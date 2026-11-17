//! Explicit finite-rate actuator receipts contracts.

use super::*;
/// Original all-position arrival window derived from the complete anchor enclosure.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorArrivalCertificate {
    /// Retained original epoch and complete position interval for this target episode.
    pub anchor: GateActuatorAnchor,
    /// Original active target in metres, inside the actual gate range.
    pub target_opening_m: f64,
    /// Original finite earliest all-position arrival bound.
    pub lower_s: f64,
    /// Original finite latest all-position arrival bound.
    pub upper_s: f64,
    /// Exact high/low expansion of the complete original arrival-window width.
    pub width_s: GateActuatorTimeExpansion,
}

/// Selected ideal response direction toward the active target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateActuatorDirection {
    /// Motion increases opening toward the active target.
    Opening,
    /// Motion decreases opening toward the active target.
    Closing,
}

/// Optional exact-product and finite TwoSum evidence for local fused rounding.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorBinaryWitness {
    /// Finite exactly represented product certified by the optional binary witness.
    pub product_m: f64,
    /// Finite TwoSum high, matching the strict fused result.
    pub sum_high_m: f64,
    /// Finite TwoSum residual giving the exact local fused rounding error.
    pub sum_low_m: f64,
}

/// One strict fused stage with original operands and independent rounding evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorFmaStageReceipt {
    /// Original selected directional rate with its exact motion sign.
    pub signed_rate_m_per_s: f64,
    /// Original elapsed high or signed low used by this fused stage.
    pub time_component_s: f64,
    /// Original anchor operand or retained primary fused result.
    pub addend_m: f64,
    /// Finite strict Numr fused multiply-add result.
    pub result_m: f64,
    /// Independent nonnegative allowance for this stage alone.
    pub rounding_error_m: f64,
    /// Whether the operand grid proves this stage's zero result mathematically exact.
    pub zero_grid_certificate: bool,
    /// Optional checked exact-product/TwoSum certificate. Absence preserves ordinary fused admission.
    pub exact_binary_witness: Option<GateActuatorBinaryWitness>,
}

/// Signed two-stage motion evaluation and uncapped position enclosure.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorMotionReceipt {
    /// Selected motion direction from the original anchor operand and target.
    pub direction: GateActuatorDirection,
    /// Original nonnegative magnitude of the selected directional rate.
    pub rate_m_per_s: f64,
    /// First fused evaluation using elapsed high and original anchor operand.
    pub primary: GateActuatorFmaStageReceipt,
    /// Second fused evaluation for nonzero signed elapsed low. None skips that operation entirely.
    pub elapsed_low: Option<GateActuatorFmaStageReceipt>,
    /// Outward sum of used fused-stage allowances, without duplicating elapsed-low correction.
    pub raw_error_m: f64,
    /// Uncapped finite lower endpoint, or None for a proved unbounded lower endpoint.
    pub raw_lower_m: Option<f64>,
    /// Uncapped finite upper endpoint, or None for a proved unbounded upper endpoint.
    pub raw_upper_m: Option<f64>,
}

/// Physical response image of one retained anchor endpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorEndpointReceipt {
    /// Original anchor endpoint evaluated by this physical response map.
    pub anchor_opening_m: f64,
    /// Motion evidence, absent for exact-time-zero, equal-target, or selected-rate-zero holds.
    pub motion: Option<GateActuatorMotionReceipt>,
    /// Certified finite lower bound on ideal actual opening, in metres.
    pub lower_opening_m: f64,
    /// Certified finite upper bound on ideal actual opening, in metres.
    pub upper_opening_m: f64,
}

/// Complete elapsed, endpoint, and center evidence for one original query.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorForwardReceipt {
    /// Retained original epoch and complete position interval for this target episode.
    pub anchor: GateActuatorAnchor,
    /// Original supported time used for this forward proof.
    pub query_time_s: f64,
    /// Original active target in metres, inside the actual gate range.
    pub target_opening_m: f64,
    /// Complete original query-minus-anchor elapsed expansion.
    pub elapsed_s: GateActuatorTimeExpansion,
    /// Physical response evidence from the retained lower anchor endpoint.
    pub lower_endpoint: GateActuatorEndpointReceipt,
    /// Physical response evidence from the retained upper anchor endpoint.
    pub upper_endpoint: GateActuatorEndpointReceipt,
    /// Center evaluation, absent when an admitted hold or completion bypasses motion.
    pub center_motion: Option<GateActuatorMotionReceipt>,
    /// Complete represented opening and inherited certified uncertainty.
    pub position: GateActuatorPosition,
}

/// Bounded analytic or forward-singleton proof for one completed active episode.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorCompletionReceipt {
    /// Zero-based active episode identity retained by this completion proof.
    pub episode_index: usize,
    /// Count of actual target changes identifying the active target bits.
    pub target_generation: usize,
    /// First accepted original time proving completion of this episode.
    pub confirmed_at_s: f64,
    /// Unchanged original analytic arrival certificate for bounded recomputation.
    pub original_arrival: GateActuatorArrivalCertificate,
    /// Minimum of original analytic upper and accepted confirmation time.
    pub effective_upper_s: f64,
    /// Exact width from original lower to the effective completion upper.
    pub effective_width_s: GateActuatorTimeExpansion,
    /// Retained forward singleton proof when confirmation precedes the analytic upper.
    pub forward: Option<GateActuatorForwardReceipt>,
}
