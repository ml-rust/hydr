//! Explicit finite-rate actuator event contracts.

/// Next strictly later event with exact coincident boundary flags.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateActuatorEvent {
    /// Original accepted or retained event time in seconds.
    pub time_s: f64,
    /// Original target-calendar index, absent when this event contains no target update.
    pub target_index: Option<usize>,
    /// Whether this event reaches a retained earliest arrival boundary.
    pub arrival_lower_boundary: bool,
    /// Whether this event reaches a retained latest arrival boundary.
    pub arrival_upper_boundary: bool,
    /// Whether this event reaches the original terminal horizon.
    pub terminal: bool,
}
