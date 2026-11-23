use super::{
    SurfaceChannelError, SurfaceChannelOperating, SurfaceChannelPoolEnd, SurfaceChannelRunoff,
    SurfaceChannelStructure,
};
use crate::channel::{ChannelEnd, ChannelInitialState, ChannelReachDefinition};
use crate::coupling::level_pools::{CoupledPoolInflowBinding, CoupledPoolSupport};
use crate::runoff::SubcatchmentState;
use crate::solver::{BoundaryFace, InitialState, SolverSetup};
use crate::storage::{LevelPoolConfig, LevelPoolElement, LevelPoolSourcePolicy, LevelPoolState};
use crate::structures::WeirApplicability;

/// Largest admitted `|hypot(reach_tangent) - 1|` for a link's unit reach tangent.
pub const SURFACE_CHANNEL_TANGENT_TOLERANCE: f64 = 1e-9;

/// Largest admitted `|W - b| / b` between a port's face width sum `W` and its end cell width `b`.
///
/// The bound covers products such as `n · dx` against a declared width.
pub const SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE: f64 = 16.0 * f64::EPSILON;

/// One reach-end port between a line of surface faces and one end of a rectangular or
/// surveyed-rectangle reach.
///
/// One hydrostatic HLL flux crosses each face. The reach end cell is the ghost neighbour of
/// every face. The coordinator owns the end: its caller forcing policy is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelPort {
    /// Unique nonempty port identity, distinct from every link identity.
    pub id: String,
    /// Exact identity of the ported setup reach. The reach must be rectangular or a surveyed
    /// reach whose section is a rectangle.
    pub reach_id: String,
    /// Reach end the port attaches to. One port binds each end at most.
    pub end: ChannelEnd,
    /// Surface faces in any order. Admission sorts them along their face line.
    ///
    /// Every face has one active surface cell. Its other side is inactive or off the grid.
    /// The faces share one axis, one face line, and one surface side, with no gap. No face is
    /// a caller open face. The face widths sum to the end cell width within
    /// [`SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE`].
    pub faces: Vec<BoundaryFace>,
    /// Evidence that the faces and the reach end share this location.
    pub mapping_evidence: String,
}

/// One lateral bank-overflow link between a surface cell and a channel cell.
///
/// Water crosses a suppressed sharp-crested rectangular weir in either direction.
/// The crest sits at or above both beds: the link never models an unobstructed connection.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelLink {
    /// Unique nonempty link identity. Synthesized channel bindings and ledgers use it.
    pub id: String,
    /// Active global row-major surface cell index.
    pub surface_cell: usize,
    /// Exact identity of the linked setup reach.
    pub reach_id: String,
    /// Ordered cell index inside the linked reach.
    pub channel_cell: usize,
    /// Crest elevation in the common vertical datum, in metres.
    ///
    /// Must be at least the surface bed and the channel bed at the linked cells.
    pub crest_elevation_m: f64,
    /// Finite positive crest length along the bank, in metres.
    pub crest_length_m: f64,
    /// Finite positive weir discharge coefficient, in SI m^(1/2)/s.
    pub weir_coefficient: f64,
    /// Coefficient, free-nappe, submergence, and reversal evidence for the weir law.
    pub applicability: WeirApplicability,
    /// Unit horizontal tangent of the reach's increasing coordinate, in surface x/y axes.
    ///
    /// Overflow into the reach carries the projection of surface velocity on this tangent.
    /// Its length must be within [`SURFACE_CHANNEL_TANGENT_TOLERANCE`] of one.
    pub reach_tangent: [f64; 2],
    /// Evidence that the surface cell and the channel cell share this bank location.
    pub mapping_evidence: String,
}

/// Named unbranched reach with its native geometry and numerical controls.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelReach {
    /// Unique nonempty reach identity used by links and initial states.
    pub id: String,
    /// Caller reach definition. Its lateral bindings must not use the `surface-exchange:` prefix.
    pub definition: ChannelReachDefinition,
}

/// Explicit convergence budgets and bounded work for common surface/channel intervals.
///
/// No default exists: every tolerance and limit is a modelling decision.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelConfig {
    /// Absolute midpoint-head iteration allowance, in metres.
    pub head_absolute_m: f64,
    /// Dimensionless relative midpoint-head iteration allowance.
    pub head_relative: f64,
    /// Absolute signed-transfer iteration allowance, in cubic metres.
    pub transfer_absolute_m3: f64,
    /// Dimensionless relative signed-transfer iteration allowance.
    pub transfer_relative: f64,
    /// Absolute physical water-residual allowance, in cubic metres.
    pub mass_absolute_m3: f64,
    /// Dimensionless relative allowance against represented storage and external physical terms.
    pub mass_relative: f64,
    /// Smallest admitted common trial duration, in seconds.
    pub minimum_step_s: f64,
    /// Largest admitted common trial duration, in seconds.
    pub maximum_step_s: f64,
    /// Maximum fixed-point iterations per attempted common interval.
    pub iteration_limit: usize,
    /// Maximum complete interval recalculations after the initial attempt.
    pub retry_limit: usize,
    /// Target fixed-point contraction per update, in `(0, 1)`.
    ///
    /// A stalled attempt retries with `1/2`, `1/4`, or `1/8` of its duration: the largest
    /// fraction whose product with the measured contraction is at most this target.
    pub contraction_target: f64,
}

impl SurfaceChannelConfig {
    /// Checks tolerances, durations, and work limits on their own.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Input`] for:
    /// - a tolerance pair that is nonfinite, negative, or zero in both parts
    /// - a step duration that is nonfinite, not positive, or out of order
    /// - a zero `iteration_limit`
    /// - a `contraction_target` outside `(0, 1)`
    pub fn check(&self) -> Result<(), SurfaceChannelError> {
        for (absolute_name, absolute, relative_name, relative) in [
            (
                "head_absolute_m",
                self.head_absolute_m,
                "head_relative",
                self.head_relative,
            ),
            (
                "transfer_absolute_m3",
                self.transfer_absolute_m3,
                "transfer_relative",
                self.transfer_relative,
            ),
            (
                "mass_absolute_m3",
                self.mass_absolute_m3,
                "mass_relative",
                self.mass_relative,
            ),
        ] {
            if !absolute.is_finite()
                || absolute < 0.0
                || !relative.is_finite()
                || relative < 0.0
                || (absolute == 0.0 && relative == 0.0)
            {
                return Err(SurfaceChannelError::input(format!(
                    "config {absolute_name}={absolute},{relative_name}={relative}: supply finite nonnegative tolerances with positive combined allowance"
                )));
            }
        }
        if !self.minimum_step_s.is_finite()
            || self.minimum_step_s <= 0.0
            || !self.maximum_step_s.is_finite()
            || self.maximum_step_s < self.minimum_step_s
        {
            return Err(SurfaceChannelError::input(format!(
                "config minimum_step_s={},maximum_step_s={}: supply finite positive durations with maximum_step_s >= minimum_step_s",
                self.minimum_step_s, self.maximum_step_s
            )));
        }
        if self.iteration_limit == 0 {
            return Err(SurfaceChannelError::input(
                "config iteration_limit=0: supply a positive iteration count",
            ));
        }
        if !(self.contraction_target > 0.0 && self.contraction_target < 1.0) {
            return Err(SurfaceChannelError::input(format!(
                "config contraction_target={}: supply a value strictly between 0 and 1",
                self.contraction_target
            )));
        }
        Ok(())
    }
}

/// Immutable surface/channel topology with link evidence and common numerical controls.
///
/// Links, ports, structures, and pool ends together are nonempty.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelSetup {
    /// Complete surface engine definition.
    pub surface: SolverSetup,
    /// Nonempty reach definitions with unique identities, in child-output order.
    pub reaches: Vec<SurfaceChannelReach>,
    /// Bank-overflow links in declaration order.
    pub links: Vec<SurfaceChannelLink>,
    /// Reach-end ports in declaration order.
    pub ports: Vec<SurfaceChannelPort>,
    /// Level pools in declaration order. Their curves use the surface vertical datum.
    ///
    /// Every pool joins at least one structure link or pool end.
    pub pools: Vec<LevelPoolElement>,
    /// One storage support declaration per pool, in pool order.
    pub pool_support: Vec<CoupledPoolSupport>,
    /// One local source policy per pool, in pool order, or `None` without pool sources.
    pub pool_sources: Option<Vec<LevelPoolSourcePolicy>>,
    /// External inflow bindings into the pools, in any order.
    pub pool_inflows: Vec<CoupledPoolInflowBinding>,
    /// Dimensional controls of the pool child. Without pools it enters no check.
    pub storage: LevelPoolConfig,
    /// Structure links between coordinator nodes, in declaration order.
    pub structures: Vec<SurfaceChannelStructure>,
    /// Reach ends joined to pools, in declaration order.
    pub pool_ends: Vec<SurfaceChannelPoolEnd>,
    /// Operating children of every dynamically owned structure, or `None` when every structure
    /// has a manual owner.
    pub operating: Option<SurfaceChannelOperating>,
    /// Native runoff owned by the coordinator, or `None` without it.
    pub runoff: Option<SurfaceChannelRunoff>,
    /// Prescribed structure geometry schedules in declaration order.
    pub geometry_schedules: Vec<crate::structures::StructureGeometrySchedule>,
    /// Prescribed pool stage-volume curve epoch schedules in declaration order.
    pub pool_curve_schedules: Vec<crate::storage::PoolCurveSchedule>,
    /// Evidence that surface cells, channel cells, and pools store no water twice.
    pub storage_exclusion_evidence: String,
    /// Explicit dimensional tolerances, common duration limits, and work limits.
    pub config: SurfaceChannelConfig,
}

/// Complete authoritative child states at one common accepted origin.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelInitialState {
    /// Finite common elapsed time, in seconds.
    ///
    /// The surface time and every reach time must equal it bit for bit.
    pub time_s: f64,
    /// Surface depth, discharge, and soil state.
    pub surface: InitialState,
    /// One `(reach id, state)` pair per setup reach, in any order.
    pub reaches: Vec<(String, ChannelInitialState)>,
    /// One accepted volume per setup pool, in setup pool order. Empty without pools.
    pub pools: Vec<LevelPoolState>,
    /// One runoff state per active modelled definition, in definition order. Empty without
    /// runoff.
    pub runoff: Vec<SubcatchmentState>,
}
