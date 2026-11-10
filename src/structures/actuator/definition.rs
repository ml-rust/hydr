//! Explicit finite-rate actuator definition contracts.

/// Explicit independent opening-error and arrival-window allowances. No defaults apply.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateActuatorAccuracy {
    /// Absolute opening-error allowance in metres, admitted independently of the relative branch.
    pub maximum_absolute_opening_error_m: f64,
    /// Relative opening-error allowance for positive openings when absolute admission does not apply.
    pub maximum_relative_opening_error: f64,
    /// Maximum exact full arrival-window width in seconds, independent of position accuracy.
    pub maximum_arrival_window_s: f64,
}

/// Immutable ideal constant asymmetric-rate response bound to one actual gate.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorDefinition {
    /// Exact actuator identity in immutable definition order.
    pub actuator_id: String,
    /// Exact identity of the bound original gate.
    pub structure_id: String,
    /// Explicit nonnegative opening speed. Zero stalls opening motion.
    pub opening_rate_m_per_s: f64,
    /// Explicit nonnegative closing speed. Zero stalls closing motion.
    pub closing_rate_m_per_s: f64,
    /// Explicit independent numerical admission policy.
    pub accuracy: GateActuatorAccuracy,
    /// Nonempty evidence identifying the supplied directional rates.
    pub rate_provenance: String,
    /// Explicit ideal-response assumptions, including excluded mechanical dynamics.
    pub ideal_response_applicability: String,
}

/// Original actual opening and target supplied explicitly at initial time.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorInitialState {
    /// Exact actuator identity in immutable definition order.
    pub actuator_id: String,
    /// Original explicitly supplied actual opening in metres, inside the bound gate range.
    pub opening_m: f64,
    /// Original active target in metres, inside the actual gate range.
    pub target_opening_m: f64,
}

/// Original finite target calendar and closed support horizon.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorSchedule {
    /// Original finite nonnegative start of supported response.
    pub initial_time_s: f64,
    /// Original strictly increasing target-event times inside the support horizon.
    pub target_times_s: Vec<f64>,
    /// Original finite terminal support time in seconds.
    pub end_s: f64,
}

/// Identity-bound causal target packet at one original calendar event.
#[derive(Clone, Debug, PartialEq)]
pub struct GateTargetUpdate {
    /// Exact actuator identity in immutable definition order.
    pub actuator_id: String,
    /// Exact identity of the bound original gate.
    pub structure_id: String,
    /// Original requested target in metres, inside the actual gate range.
    pub target_opening_m: f64,
    /// Original issue time, exactly equal to the target event.
    pub issued_at_s: f64,
    /// Finite availability time no later than issue time. Negative timestamps remain valid.
    pub available_at_s: f64,
}
