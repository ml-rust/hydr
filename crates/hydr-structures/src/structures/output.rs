//! Explicit physical regime and proposal-only output labels.

use super::{DirectionPolicy, FlowDirection, StructureHeads};

/// Concrete evaluated constitutive model identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureLawKind {
    /// HDS-5 culvert under inlet or outlet control.
    Culvert,
    /// Hydrostatic pressure-strip integral across a thin rectangular aperture.
    HydrostaticRectangularGate,
    /// Fixed-orientation bounded pump reference curve.
    Pump,
    /// Fully submerged opening.
    SubmergedOrifice,
    /// Suppressed sharp-crested rectangular overflow with adopted Villemonte correction.
    SuppressedRectangularWeir,
}
/// Physical classification before ideal flap blocking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureRegime {
    /// Explicit zero opening isolates the endpoints without evaluating hydraulic heads.
    GateClosed,
    /// No supported wet donor lies above the gate sill.
    GateBelowSill,
    /// The wetted aperture carries only an atmospheric free strip.
    FreeGateAperture,
    /// The wetted aperture contains both submerged and atmospheric free strips.
    MixedGateAperture,
    /// Both wet heads cover the complete original opening.
    SubmergedGateAperture,
    /// Evaluated supported running pump query, including declared zero-flow support.
    PumpCurve,
    /// Explicit stopped isolation makes hydraulic heads uninfluential.
    PumpIsolated,
    /// Eligible fully submerged unequal-head opening.
    SubmergedOrifice,
    /// Eligible rectangular overflow without submerged tailwater.
    FreeWeirOverflow,
    /// Eligible rectangular overflow with positive tailwater above crest.
    SubmergedWeirOverflow,
    /// No supported wet donor above the weir crest.
    BelowCrest,
    /// Eligible wet stages are exactly equal above the controlling elevation.
    EqualHeads,
    /// Culvert discharge certainly follows the inlet-control headwater.
    CulvertInletControl,
    /// Culvert discharge certainly follows the outlet-control headwater.
    CulvertOutletControl,
    /// Both control discharges overlap within their enclosures, so either can govern.
    CulvertControlTie,
    /// No wet donor stands above its culvert invert.
    CulvertBelowInvert,
    /// The heads drive flow into an end without a declared culvert inlet.
    CulvertUndeclaredEnd,
    /// The element's condition closes the structure. Exact zero discharge under every head.
    Closed,
}
/// Proposal status, without donor availability or transfer acceptance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureStatus {
    /// Running positive proposal in the immutable pump orientation.
    PumpRunning,
    /// Supported running query in the declared zero-flow region.
    PumpShutoff,
    /// Explicit isolation excludes forward siphoning and reverse flow.
    PumpStoppedIsolated,
    /// Nonzero constitutive law evaluated in the proposed direction.
    Active,
    /// Explicit physical zero, classified by the separate regime.
    PhysicalZero,
    /// Supported proposed direction is blocked by the declared ideal flap.
    FlapBlocked,
}
/// Original evaluated law operands, including running pump shutoff. Isolated branches expose no fabricated heads.
#[derive(Clone, Debug, PartialEq)]
pub enum StructureHydraulicOperands {
    /// Culvert control heads above the bed and the certified per-barrel discharge.
    Culvert {
        /// Donor stage above the inlet bed elevation (invert without sediment), in metres.
        headwater_m: f64,
        /// Receiver stage above the outlet bed elevation (invert without sediment), in metres. Zero for a free outfall.
        tailwater_m: f64,
        /// `(lower, upper)` enclosure of the discharge of one barrel, in m³/s.
        barrel_discharge_m3_per_s: (f64, f64),
    },
    /// Evaluated hydrostatic-aperture geometry and independent arithmetic witnesses.
    HydrostaticRectangularGate(super::GateHydraulicOperands),
    /// Evaluated running/shutoff pump query and full normalization evidence.
    Pump(super::PumpHydraulicOperands),
    /// Fully submerged orifice differential-head operands.
    SubmergedOrifice {
        /// Positive difference of original supported wet stages, in metres.
        head_gap_m: f64,
        /// Natural logarithm of the proposed SI discharge magnitude.
        log_discharge_magnitude: f64,
    },
    /// Free suppressed-weir overflow operands.
    FreeWeir {
        /// Positive original donor-stage height above crest, in metres.
        upstream_head_m: f64,
        /// Natural logarithm of the proposed SI discharge magnitude.
        log_discharge_magnitude: f64,
    },
    /// Submerged suppressed-weir overflow operands.
    SubmergedWeir {
        /// Positive original donor-stage height above crest, in metres.
        upstream_head_m: f64,
        /// Positive original receiver-stage height above crest, in metres.
        downstream_head_m: f64,
        /// Independently positive direct original-stage difference, in metres.
        head_gap_m: f64,
        /// Represented Hd/Hu ratio, including rounded endpoints zero/one.
        submergence_ratio: f64,
        /// Stable factor 1+r/(1+sqrt(r)).
        correction_k: f64,
        /// Natural logarithm of the proposed SI discharge magnitude.
        log_discharge_magnitude: f64,
    },
}
/// Immutable instantaneous proposal. This quantity is never an accepted or availability-limited transfer.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureOutput {
    /// Original passive or explicit device operation, held during proposal evaluation.
    pub operation: super::StructureOperatingPoint,
    /// Ordered element index.
    pub element: usize,
    /// Complete original endpoint support and declared identities.
    pub input: StructureHeads,
    /// Concrete coefficient/law identity within the immutable batch definition.
    pub law: StructureLawKind,
    /// Immutable ideal flap/direction declaration.
    pub direction_policy: DirectionPolicy,
    /// Physical regime before flap blocking.
    pub regime: StructureRegime,
    /// Active, physical-zero, or explicitly blocked proposal.
    pub status: StructureStatus,
    /// Nominal physical direction, retained for blocked flows.
    pub proposed_direction: Option<FlowDirection>,
    /// Proposed signed discharge in m³/s. Positive means A to B.
    pub discharge_m3_per_s: f64,
    /// Evaluated operands, absent for physical zeros, flap blocking, and isolated operations.
    pub hydraulic_operands: Option<StructureHydraulicOperands>,
}

/// Exact elapsed support of a held-head proposal, without an evolving trajectory claim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeldHeadInterval {
    /// Finite nonnegative elapsed start in seconds.
    pub start: f64,
    /// Finite strictly later elapsed end in seconds.
    pub end: f64,
}
/// Original held-head signed Q*dt proposal, without donor or paired-component admission.
#[derive(Clone, Debug, PartialEq)]
pub struct HeldHeadVolumeOutput {
    /// Exact original interval.
    pub interval: HeldHeadInterval,
    /// Immutable original instantaneous proposal.
    pub proposal: StructureOutput,
    /// Original signed discharge-times-duration product in m³.
    pub volume_m3: f64,
}
