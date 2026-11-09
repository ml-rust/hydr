//! Explicit physical regime and proposal-only output labels.

use super::{DirectionPolicy, FlowDirection, StructureHeads};

/// Concrete evaluated constitutive model identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureLawKind {
    /// Fully submerged opening.
    SubmergedOrifice,
    /// Suppressed sharp-crested rectangular overflow with adopted Villemonte correction.
    SuppressedRectangularWeir,
}
/// Physical classification before ideal flap blocking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureRegime {
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
}
/// Proposal status, without donor availability or transfer acceptance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructureStatus {
    /// Nonzero constitutive law evaluated in the proposed direction.
    Active,
    /// Explicit physical zero, classified by the separate regime.
    PhysicalZero,
    /// Supported proposed direction is blocked by the declared ideal flap.
    FlapBlocked,
}
/// Original active law operands. Inactive branches expose no fabricated numerical heads.
#[derive(Clone, Debug, PartialEq)]
pub enum StructureHydraulicOperands {
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
    /// Checked active law operands, absent for physical-zero/blocked branches.
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
