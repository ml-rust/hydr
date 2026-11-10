//! Explicit causal sampled hydraulic support.

/// External stage support declaration without inferred endpoint attachment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ControlStageSignal {
    /// Unique signal identity within this controller.
    pub signal_id: String,
    /// External hydraulic support identity.
    pub support_id: String,
    /// Exact common vertical reference.
    pub datum: String,
}
/// Original observed stage packet for one controller decision.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlSample {
    /// Original controller identity.
    pub controller_id: String,
    /// Declared signal identity.
    pub signal_id: String,
    /// Declared external hydraulic support.
    pub support_id: String,
    /// Exact common vertical reference.
    pub datum: String,
    /// Original observation time in seconds.
    pub observed_at_s: f64,
    /// Original availability time in seconds, no later than observation.
    pub available_at_s: f64,
    /// Finite observed stage in metres.
    pub stage_m: f64,
}
