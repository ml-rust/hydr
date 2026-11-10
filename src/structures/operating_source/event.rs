//! Exact component event origins without dynamic actuator-lane projection.

/// Next strictly later supported component event or common terminal bound.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatingSourceEvent {
    /// Original represented event time in seconds. No epsilon merging applies.
    pub time_s: f64,
    /// The event is the registry's common terminal bound.
    pub terminal: bool,
    /// Exactly coincident origins in original source registration order.
    pub origins: Vec<OperatingSourceEventOrigin>,
}
/// Static registered source identity and its component event summary.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatingSourceEventOrigin {
    /// Original registered source identity.
    pub source_id: String,
    /// Exact component event kind, without inferred actuator IDs.
    pub kind: OperatingSourceEventKind,
}
/// Supported point-source event semantics. This predicts no complete future feedback sequence.
#[derive(Clone, Debug, PartialEq)]
pub enum OperatingSourceEventKind {
    /// A prescribed source's original knot event.
    PrescribedKnot,
    /// A fresh actuator frame's scalar event summary.
    Actuator {
        /// Original target index when this event installs complete target packets.
        target_index: Option<usize>,
        /// At least one current arrival window reaches its lower boundary.
        arrival_lower_boundary: bool,
        /// At least one current arrival window reaches its upper boundary.
        arrival_upper_boundary: bool,
        /// This actuator reaches its original terminal support.
        terminal: bool,
    },
}
