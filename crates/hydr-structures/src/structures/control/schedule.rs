//! Original finite control support and explicit sampled timing policy.

/// Complete original decision schedule without implicit extension.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlDecisionSchedule {
    /// Nonnegative accepted start in seconds.
    pub initial_time_s: f64,
    /// Strictly increasing decisions after the start, at or before the horizon.
    pub decision_times_s: Vec<f64>,
    /// Inclusive finite support horizon in seconds.
    pub end_s: f64,
}
/// Caller-declared sampled applicability and timing bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledControlTiming {
    /// Maximum exact observation/decision spacing in seconds.
    pub maximum_decision_gap_s: f64,
    /// Maximum qualification-expiry lateness in seconds.
    pub maximum_timer_lateness_s: f64,
    /// Evidence that this sampled policy fits the intended use.
    pub sampled_applicability: String,
}
/// Next original decision, terminal boundary, or both.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlEvent {
    /// Original event time in seconds.
    pub time_s: f64,
    /// Original decision index, absent at a separate terminal boundary.
    pub decision_index: Option<usize>,
    /// Event equals the declared support horizon.
    pub terminal: bool,
}
