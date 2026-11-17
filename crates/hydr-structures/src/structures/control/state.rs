//! Complete explicit relay continuation and assumed antecedent state.

use super::{ControlLatch, ControlSample, ControlTransitionReceipt};

/// Opposing sampled qualification anchored at an original time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlPending {
    /// Latch reached when qualification completes.
    pub target: ControlLatch,
    /// Original qualification start in seconds.
    pub since_s: f64,
}
/// Explicit initial memory. Omitted antecedents declare fresh memory without historical proof.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlInitialState {
    /// Original ordered controller identity.
    pub controller_id: String,
    /// Explicit initial relay latch.
    pub latch: ControlLatch,
    /// Assumed incomplete opposing qualification.
    pub pending: Option<ControlPending>,
    /// Last declared causal observation, if supplied.
    pub antecedent_sample: Option<ControlSample>,
    /// Last assumed transition time, if supplied.
    pub last_transition_at_s: Option<f64>,
}
/// Complete projected continuation for one controller.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlState {
    /// Original ordered controller identity.
    pub controller_id: String,
    /// Current relay latch.
    pub latch: ControlLatch,
    /// Current incomplete sampled qualification.
    pub pending: Option<ControlPending>,
    /// Last accepted causal observation.
    pub last_sample: Option<ControlSample>,
    /// Last assumed or in-run transition time.
    pub last_transition_at_s: Option<f64>,
    /// Exact transitions since this batch's explicit initial state.
    pub transition_count: usize,
    /// Complete last in-run transition, absent before the first one.
    pub last_transition: Option<ControlTransitionReceipt>,
}
