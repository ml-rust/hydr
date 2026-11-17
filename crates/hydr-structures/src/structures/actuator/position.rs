//! Explicit finite-rate actuator position contracts.

/// Finite represented opening and certified enclosure of the ideal response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateActuatorPosition {
    /// Represented actual opening in metres, inside its certified enclosure.
    pub opening_m: f64,
    /// Certified finite lower bound on ideal actual opening, in metres.
    pub lower_opening_m: f64,
    /// Certified finite upper bound on ideal actual opening, in metres.
    pub upper_opening_m: f64,
    /// Finite bound covering both center-to-endpoint distances, in metres.
    pub absolute_error_m: f64,
}

/// Original target-episode epoch and complete inherited position uncertainty.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateActuatorAnchor {
    /// Original accepted or retained event time in seconds.
    pub time_s: f64,
    /// Complete represented opening and inherited certified uncertainty.
    pub position: GateActuatorPosition,
}

/// Exact finite time difference represented by high and signed low components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateActuatorTimeExpansion {
    /// Rounded high component of the original time difference.
    pub high_s: f64,
    /// Signed finite TwoSum residual completing the exact time difference.
    pub low_s: f64,
}

/// Arrival classification relative to retained completion and support certificates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateActuatorPhase {
    /// Accepted time precedes the certified earliest arrival boundary.
    BeforeWindow,
    /// Accepted time lies inside the unresolved finite arrival window.
    InsideWindow,
    /// A retained analytic or forward-singleton proof certifies all-position completion.
    Complete,
    /// A lower certificate proves no arrival within remaining support.
    BeyondHorizon,
    /// Every admitted position lies strictly on the same zero-rate side.
    Stalled,
    /// Some enclosed positions cannot reach target while others can arrive or already equal it.
    UnresolvedWithStalledSupport,
}

/// Certified opening support, independent of rounded center equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateActuatorSupport {
    /// Canonical zero opening, zero singleton enclosure, and zero error.
    ExactClosed,
    /// The certified lower opening bound is strictly positive.
    CertainlyPositive,
    /// The certified interval includes both zero and positive support.
    ClosureUnresolved,
}

/// Explicit current position, support, and effective arrival-window projection.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorOutput {
    /// Index in the complete original structure context, never a compact subset index.
    pub element: usize,
    /// Exact actuator identity in immutable definition order.
    pub actuator_id: String,
    /// Exact identity of the bound original gate.
    pub structure_id: String,
    /// Original accepted or retained event time in seconds.
    pub time_s: f64,
    /// Original active target in metres, inside the actual gate range.
    pub target_opening_m: f64,
    /// Complete represented opening and inherited certified uncertainty.
    pub position: GateActuatorPosition,
    /// Current arrival/completion classification within declared support.
    pub phase: GateActuatorPhase,
    /// Certified closed, positive, or unresolved opening support.
    pub support: GateActuatorSupport,
    /// Effective finite earliest arrival bound, absent without a finite window.
    pub arrival_lower_s: Option<f64>,
    /// Effective finite latest arrival bound, tightened only by certified completion.
    pub arrival_upper_s: Option<f64>,
}
