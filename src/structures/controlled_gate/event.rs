//! Exact common decisions, arrival boundaries, and finite terminal support.

/// Next strictly later common event with exact coincident flags.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ControlledGateEvent {
    /// Original represented event time in seconds.
    pub time_s: f64,
    /// Original simultaneous decision index, absent at arrival-only boundaries.
    pub decision_index: Option<usize>,
    /// Event reaches a retained earliest actuator arrival boundary.
    pub arrival_lower_boundary: bool,
    /// Event reaches a retained latest actuator arrival boundary.
    pub arrival_upper_boundary: bool,
    /// Event reaches the original common terminal horizon.
    pub terminal: bool,
}
