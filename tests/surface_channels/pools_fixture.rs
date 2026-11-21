//! Pool, structure, pool-end, and operating fixtures through the public coordinator API.
//!
//! One 2 m square surface cell sits beside one 2 m by 2 m walled reach cell. Both hold `AREA`
//! of plan area behind impermeable walls. Constant-area ponds join them through structure links
//! and pool ends. Operating children are declared as data.

use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelEnd, ChannelGeometry, ChannelInitialState,
    ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput, ChannelReachDefinition,
    ChannelSurveyedConfig, CrossSectionDefinition, CrossSectionPoint, RectangularChannelCell,
    SurveyedChannelCell,
};
#[path = "pools_laws.rs"]
pub mod laws;

pub use laws::{
    controlled_gate, controlled_pump, culvert_law, gate_law, operating, orifice_law, pump_law,
    signal, trajectory, weir_law,
};

use hydr::coupling::{
    CoupledPoolInflow, CoupledPoolInflowBinding, CoupledPoolSupport, StructureScenario,
    SurfaceChannelAttachment, SurfaceChannelBatch, SurfaceChannelConfig,
    SurfaceChannelForcingInterval, SurfaceChannelInitialState, SurfaceChannelLink,
    SurfaceChannelNode, SurfaceChannelOperating, SurfaceChannelOperationOwner,
    SurfaceChannelPoolEnd, SurfaceChannelReach, SurfaceChannelReachForcing, SurfaceChannelSetup,
    SurfaceChannelStructure, SurfaceChannelStructureTransfer,
};
use hydr::forcing::WaterAmount;
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, SolverConfig, SolverSetup,
};
use hydr::storage::{
    LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolKnot, LevelPoolSourcePolicy,
    LevelPoolSources, LevelPoolState,
};
use hydr::structures::{
    DirectionPolicy, StructureElement, StructureLaw, StructureOperatingPoint, WeirApplicability,
};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Batch = SurfaceChannelBatch<crate::runtime::Runtime>;

pub const DATUM: &str = "Synthetic structure datum";
pub const REACH: &str = "r";
/// Plan area of the surface cell and of the reach cell, in m².
pub const AREA: f64 = 4.0;
/// Caller inflow binding on the reach cell.
pub const INFLOW: &str = "caller:inflow";
/// Caller withdrawal binding on the reach cell.
pub const WITHDRAWAL: &str = "caller:withdrawal";

/// The surface cell.
pub fn surface() -> SurfaceChannelNode {
    SurfaceChannelNode::SurfaceCell(0)
}

/// The reach cell.
pub fn channel() -> SurfaceChannelNode {
    SurfaceChannelNode::ChannelCell {
        reach_id: REACH.into(),
        cell: 0,
    }
}

/// Node of pool `id`.
pub fn pool(id: &str) -> SurfaceChannelNode {
    SurfaceChannelNode::Pool(id.into())
}

/// A manual owner holding `operation`.
pub fn manual(operation: StructureOperatingPoint) -> SurfaceChannelOperationOwner {
    SurfaceChannelOperationOwner::Manual(operation)
}

/// Structure `id` from node `a` to node `b`, each with its attachment bed.
pub fn structure(
    id: &str,
    law: StructureLaw,
    direction: DirectionPolicy,
    operation: SurfaceChannelOperationOwner,
    (a, bed_a): (SurfaceChannelNode, f64),
    (b, bed_b): (SurfaceChannelNode, f64),
) -> SurfaceChannelStructure {
    SurfaceChannelStructure {
        element: StructureElement {
            id: id.into(),
            endpoint_a: a.identity(),
            endpoint_b: b.identity(),
            datum: DATUM.into(),
            direction,
            law,
        },
        endpoint_a: SurfaceChannelAttachment {
            node: a,
            bed_elevation_m: bed_a,
            physical_connection_assertion: "The structure opens onto endpoint A".into(),
        },
        endpoint_b: SurfaceChannelAttachment {
            node: b,
            bed_elevation_m: bed_b,
            physical_connection_assertion: "The structure opens onto endpoint B".into(),
        },
        scenario: StructureScenario {
            id: format!("{id}-as-built"),
            evidence: "As-built geometry, no blockage".into(),
        },
        operation,
    }
}

/// A constant-area pond with its origin at `origin`, `area` m² of plan area, and `height` m of
/// curve above the origin.
pub fn pond(id: &str, origin: f64, area: f64, height: f64) -> Result<LevelPoolElement> {
    let knot = |volume_m3, stage_m| LevelPoolKnot { volume_m3, stage_m };
    Ok(LevelPoolElement {
        id: id.into(),
        curve: LevelPoolCurve::new(
            vec![knot(0.0, origin), knot(area * height, origin + height)],
            DATUM.into(),
            "Synthetic constant-area pond".into(),
        )?,
        connected_pool_assertion: "One level water body".into(),
        storage_support_identity: format!("{id}-basin"),
    })
}

/// Pool end `id` from `end` of the tank reach into pool `pool`.
pub fn pool_end(id: &str, end: ChannelEnd, pool: &str) -> SurfaceChannelPoolEnd {
    SurfaceChannelPoolEnd {
        id: id.into(),
        reach_id: REACH.into(),
        end,
        pool_id: pool.into(),
        entrance_loss_coefficient: 0.5,
        physical_connection_assertion: "The reach outfall opens into the pond at its bed".into(),
        quiescent_reservoir_assertion: "The pond is quiescent and mixes the outfall water".into(),
    }
}

/// External inflow binding `id` into pool `pool`.
pub fn inflow_binding(id: &str, pool: &str) -> CoupledPoolInflowBinding {
    CoupledPoolInflowBinding {
        boundary_id: id.into(),
        pool_id: pool.into(),
        input_id: format!("{id}-input"),
        physical_support_id: format!("{id}-pipe"),
        boundary_assertion: "The pipe discharges into the pond".into(),
        ownership_assertion: "No other model holds this water".into(),
    }
}

/// External inflow input of binding `id`.
pub fn inflow_input(id: &str, value: WaterAmount) -> CoupledPoolInflow {
    CoupledPoolInflow {
        boundary_id: id.into(),
        input_id: format!("{id}-input"),
        value,
    }
}

/// A bank weir between the surface cell and the reach cell, crest at `crest`.
pub fn bank(id: &str, crest: f64) -> SurfaceChannelLink {
    SurfaceChannelLink {
        id: id.into(),
        surface_cell: 0,
        reach_id: REACH.into(),
        channel_cell: 0,
        crest_elevation_m: crest,
        crest_length_m: 1.0,
        weir_coefficient: 1.7,
        applicability: WeirApplicability {
            coefficient_provenance: "Declared weir coefficient".into(),
            free_nappe_and_approach_assertion: "Suppressed crest, quiescent approach".into(),
            submergence_correction_assertion: "Villemonte correction adopted".into(),
            symmetric_reversal: true,
        },
        reach_tangent: [1.0, 0.0],
        mapping_evidence: "The surface cell and the reach cell share this bank".into(),
    }
}

/// The rectangular tank reach on bed `bed`.
pub fn tank_reach(bed: f64) -> ChannelReachDefinition {
    let binding = |boundary_id: &str, direction| ChannelLateralBinding {
        boundary_id: boundary_id.into(),
        cell_id: "cell-0".into(),
        direction,
        ownership_assertion: "Caller water at the reach cell".into(),
    };
    ChannelReachDefinition {
        vertical_datum: DATUM.into(),
        provenance: "One walled rectangular tank cell".into(),
        geometry: ChannelGeometry::Rectangular {
            cells: vec![RectangularChannelCell {
                id: "cell-0".into(),
                length_m: 2.0,
                width_m: 2.0,
                bed_elevation_m: bed,
                maximum_depth_m: 4.0,
                manning_n: 0.0,
            }],
        },
        lateral_bindings: vec![
            binding(INFLOW, ChannelLateralDirection::Inflow),
            binding(WITHDRAWAL, ChannelLateralDirection::Withdrawal),
        ],
        config: ChannelConfig::default(),
    }
}

/// The tank reach as a surveyed rectangular section on bed `bed`.
pub fn surveyed_reach(bed: f64) -> ChannelReachDefinition {
    let rectangular = tank_reach(bed);
    ChannelReachDefinition {
        geometry: ChannelGeometry::PrismaticSurveyed {
            profile: CrossSectionDefinition {
                id: "section".into(),
                vertical_datum: DATUM.into(),
                points: [(0.0, 4.0), (0.0, bed), (2.0, bed), (2.0, 4.0)]
                    .into_iter()
                    .map(|(station_m, elevation_m)| CrossSectionPoint {
                        station_m,
                        elevation_m,
                    })
                    .collect(),
                maximum_stage_m: 4.0,
                profile_provenance: "Analytical rectangular survey".into(),
            },
            cells: vec![SurveyedChannelCell {
                id: "cell-0".into(),
                length_m: 2.0,
                manning_n: 0.0,
            }],
            numerical: ChannelSurveyedConfig::default(),
        },
        ..rectangular
    }
}

/// One coordinator case: the tanks, ponds at their volumes, and every coupling declaration.
#[derive(Clone, Debug)]
pub struct Case {
    pub surface_bed: f64,
    pub surface_stage: f64,
    pub reach_bed: f64,
    pub reach_stage: f64,
    /// Ponds in declaration order with their initial volumes, in m³.
    pub ponds: Vec<(LevelPoolElement, f64)>,
    /// Declare one local source policy per pond.
    pub sources: bool,
    pub inflows: Vec<CoupledPoolInflowBinding>,
    pub links: Vec<SurfaceChannelLink>,
    pub structures: Vec<SurfaceChannelStructure>,
    pub pool_ends: Vec<SurfaceChannelPoolEnd>,
    pub operating: Option<SurfaceChannelOperating>,
}

impl Case {
    /// Tanks on beds at `0.0` with stages `surface` and `reach`, and nothing else.
    pub fn tanks(surface: f64, reach: f64) -> Self {
        Self {
            surface_bed: 0.0,
            surface_stage: surface,
            reach_bed: 0.0,
            reach_stage: reach,
            ponds: Vec::new(),
            sources: false,
            inflows: Vec::new(),
            links: Vec::new(),
            structures: Vec::new(),
            pool_ends: Vec::new(),
            operating: None,
        }
    }

    /// Setup and initial state of this case.
    pub fn parts(&self) -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
        let grid = CartesianGrid::new(1, 1, 2.0, 2.0)?;
        let domain = Domain::new(
            grid,
            vec![self.surface_bed],
            vec![0.0],
            vec![true],
            SpatialReference::new("Synthetic projected CRS".into(), DATUM.into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 1000.0,
                rain: vec![0.0],
                potential_evaporation: vec![0.0],
                withdrawal: vec![0.0],
                boundaries: Vec::new(),
            }],
        )?;
        let land = LandModel::impervious(&domain);
        let soil = SoilState::initialized(&land)?;
        let flow = FlowState::new(
            &domain,
            vec![CellState {
                h: (self.surface_stage - self.surface_bed).max(0.0),
                qx: 0.0,
                qy: 0.0,
            }],
        )?;
        let pools: Vec<LevelPoolElement> = self.ponds.iter().map(|(p, _)| p.clone()).collect();
        let support = pools
            .iter()
            .map(|pool| CoupledPoolSupport {
                pool_id: pool.id.clone(),
                zero_volume_stage_m: pool.curve.origin(),
                zero_volume_limit_assertion: "The pond empties at its curve origin".into(),
            })
            .collect();
        let sources = self.sources.then(|| {
            pools
                .iter()
                .map(|_| LevelPoolSourcePolicy {
                    dry_basal_footprint_assertion: "The first segment is the rain footprint".into(),
                    bottom_loss_provenance: "Supplied bottom demand".into(),
                })
                .collect()
        });
        let setup = SurfaceChannelSetup {
            surface: SolverSetup {
                domain,
                land,
                boundaries,
                forcing,
                config: SolverConfig::default(),
            },
            reaches: vec![SurfaceChannelReach {
                id: REACH.into(),
                definition: tank_reach(self.reach_bed),
            }],
            links: self.links.clone(),
            ports: Vec::new(),
            pools,
            pool_support: support,
            pool_sources: sources,
            pool_inflows: self.inflows.clone(),
            storage: LevelPoolConfig::default(),
            structures: self.structures.clone(),
            pool_ends: self.pool_ends.clone(),
            operating: self.operating.clone(),
            runoff: None,
            storage_exclusion_evidence: "The reach tank and the ponds lie outside the surface cell"
                .into(),
            config: SurfaceChannelConfig {
                head_absolute_m: 1e-12,
                head_relative: 1e-12,
                transfer_absolute_m3: 1e-12,
                transfer_relative: 1e-12,
                mass_absolute_m3: 1e-9,
                mass_relative: 1e-12,
                minimum_step_s: 1e-10,
                maximum_step_s: 1.0,
                iteration_limit: 64,
                retry_limit: 16,
                contraction_target: 0.5,
            },
        };
        let initial = SurfaceChannelInitialState {
            time_s: 0.0,
            surface: InitialState {
                time: 0.0,
                flow,
                soil,
            },
            reaches: vec![(
                REACH.into(),
                ChannelInitialState {
                    time_s: 0.0,
                    volume_m3: vec![(self.reach_stage - self.reach_bed).max(0.0) * AREA],
                    longitudinal_momentum_m4_per_s: vec![0.0],
                },
            )],
            pools: self
                .ponds
                .iter()
                .map(|&(_, volume_m3)| LevelPoolState { volume_m3 })
                .collect(),
            runoff: Vec::new(),
        };
        Ok((setup, initial))
    }

    /// The batch of this case.
    pub fn build(&self) -> Result<Batch> {
        let (setup, initial) = self.parts()?;
        Ok(Batch::new(crate::runtime::client(), setup, initial)?)
    }
}

/// Pool forcing of one interval: source rates per pond and inputs per inflow binding.
#[derive(Clone, Debug, Default)]
pub struct PoolForcing {
    pub sources: Option<Vec<LevelPoolSources>>,
    pub inflows: Vec<CoupledPoolInflow>,
}

/// Forcing toward `end` with zero caller lateral rates, walls at caller-owned reach ends, and
/// `pools`.
pub fn forcing(batch: &Batch, end: f64, pools: &PoolForcing) -> SurfaceChannelForcingInterval {
    let owned = |side: ChannelEnd| batch.setup().pool_ends.iter().any(|e| e.end == side);
    let policy = |side| (!owned(side)).then_some(ChannelBoundary::Wall);
    let rate = |rate_m3_per_s| WaterAmount::Rate { rate_m3_per_s };
    SurfaceChannelForcingInterval {
        start_s: batch.time(),
        end_s: end,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: policy(ChannelEnd::Left),
            right: policy(ChannelEnd::Right),
            lateral_inputs: vec![
                ChannelLateralInput::Inflow {
                    boundary_id: INFLOW.into(),
                    amount: rate(0.0),
                    longitudinal_velocity_m_per_s: 0.0,
                },
                ChannelLateralInput::Withdrawal {
                    boundary_id: WITHDRAWAL.into(),
                    amount: rate(0.0),
                },
            ],
        }],
        pool_sources: pools.sources.clone(),
        pool_inflows: pools.inflows.clone(),
        runoff: Vec::new(),
    }
}

/// One accepted interval toward `batch.time() + dt` under `pools`.
pub fn step(batch: &Batch, dt: f64, pools: &PoolForcing) -> Result<Batch> {
    let end = batch.time() + dt;
    Ok(batch
        .trial_to(end, &forcing(batch, end, pools))?
        .into_next())
}

/// The receipt of structure `id` over the last accepted interval.
pub fn transfer(batch: &Batch, id: &str) -> Result<SurfaceChannelStructureTransfer> {
    let step = batch.last_step()?.ok_or("an accepted interval")?;
    Ok(step
        .structures
        .into_iter()
        .find(|t| t.structure_id == id)
        .ok_or("the structure receipt")?)
}

/// Debug text of the package, every ledger, and every live child state.
///
/// Debug text shows every bit of finite values, signed zero included.
pub fn state(batch: &Batch) -> Result<String> {
    let pools = batch.pools().map(|pools| pools.states()).transpose()?;
    Ok(format!(
        "{} {} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
        batch.time(),
        batch.accepted_steps(),
        batch.snapshot_data()?,
        batch.ledger()?,
        batch.structure_ledgers()?,
        batch.pool_end_ledgers()?,
        batch.pool_inflow_ledgers()?,
        batch.link_ledgers()?,
        pools,
        batch.surface().state()?
    ))
}
