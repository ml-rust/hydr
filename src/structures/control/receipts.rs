//! Finite original-time witnesses and bounded decision diagnostics.

use super::{ControlCommandOutput, ControlLatch, ControlSample, ControlState};

/// Exact finite two-component elapsed-time expansion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlTimeExpansion {
    /// Rounded high component in seconds.
    pub high_s: f64,
    /// Exact residual component in seconds.
    pub low_s: f64,
}
/// Original timer operands and checked readiness/lateness witnesses.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlTimerWitness {
    /// Original timer start in seconds.
    pub started_at_s: f64,
    /// Original decision time in seconds.
    pub evaluated_at_s: f64,
    /// Original directional qualification delay in seconds.
    pub delay_s: f64,
    /// Rounded elapsed high in seconds.
    pub elapsed_high_s: f64,
    /// Exact elapsed residual in seconds.
    pub elapsed_low_s: f64,
    /// Exact elapsed time reaches the original delay.
    pub ready: bool,
    /// Exact finite delay-plus-lateness expansion, absent on positive high overflow.
    pub allowed_elapsed: Option<ControlTimeExpansion>,
    /// Positive rounded overflow of the allowed elapsed threshold.
    pub allowed_elapsed_overflow: bool,
    /// Elapsed time fits the declared lateness allowance.
    pub within_lateness: bool,
}
/// Complete retained last transition without recursive prior-state history.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlTransitionReceipt {
    /// Original scheduled decision index.
    pub decision_index: usize,
    /// Original decision time in seconds.
    pub time_s: f64,
    /// Prior latch.
    pub from: ControlLatch,
    /// Resulting opposing latch.
    pub to: ControlLatch,
    /// Original causal observation.
    pub sample: ControlSample,
    /// Complete qualification witness.
    pub timer: ControlTimerWitness,
}
/// One simultaneous decision's original before-state and computed after-state.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlDecisionReceipt {
    /// Complete structure-batch index.
    pub element: usize,
    /// Original controller identity.
    pub controller_id: String,
    /// Complete immediate before-state without a prior-step link.
    pub before: ControlState,
    /// Original causal observation.
    pub sample: ControlSample,
    /// Current sample meets the opposing threshold.
    pub qualifies: bool,
    /// Used qualification timer, absent when the predicate is false.
    pub timer: Option<ControlTimerWitness>,
    /// This decision changes the latch.
    pub transitioned: bool,
    /// Complete resulting continuation.
    pub after: ControlState,
}
/// Complete last interval diagnostics with bounded retained state.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlStepDiagnostics {
    /// Accepted interval start in seconds.
    pub start_s: f64,
    /// Accepted interval end in seconds.
    pub end_s: f64,
    /// Original decision index, absent for a non-decision interval.
    pub decision_index: Option<usize>,
    /// Commands applying to the interval beginning at its start.
    pub left_commands: Vec<ControlCommandOutput>,
    /// Commands applying to the interval beginning at its end.
    pub right_commands: Vec<ControlCommandOutput>,
    /// Simultaneous decision receipts, empty without a decision.
    pub decisions: Vec<ControlDecisionReceipt>,
}
