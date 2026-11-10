//! Explicit finite-rate actuator state contracts.

use super::*;
/// Bounded old-target left-state evidence and changed-target activation.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorTargetChangeReceipt {
    /// Original target-calendar index associated with this event or change.
    pub target_index: usize,
    /// Original accepted or retained event time in seconds.
    pub time_s: f64,
    /// Original active target before this actual change.
    pub previous_target_opening_m: f64,
    /// Complete old-target anchor retained for left-state recomputation.
    pub previous_anchor: GateActuatorAnchor,
    /// Bounded prior completion proof, retained independently of newer episodes.
    pub previous_completion: Option<GateActuatorCompletionReceipt>,
    /// Original causal packet installing this changed target.
    pub update: GateTargetUpdate,
    /// Complete old-target endpoint position before any target change.
    pub left_position: GateActuatorPosition,
    /// Changed-target anchor carrying the complete admitted left-position interval.
    pub new_anchor: GateActuatorAnchor,
    /// Count of actual target changes identifying the active target bits.
    pub target_generation: usize,
    /// New active episode identity, absent when the changed target begins an exact hold.
    pub new_episode_index: Option<usize>,
}

/// Complete current continuation with bounded target and completion receipts.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorState {
    /// Exact actuator identity in immutable definition order.
    pub actuator_id: String,
    /// Retained original epoch and complete position interval for this target episode.
    pub anchor: GateActuatorAnchor,
    /// Original active target in metres, inside the actual gate range.
    pub target_opening_m: f64,
    /// Complete represented opening and inherited certified uncertainty.
    pub position: GateActuatorPosition,
    /// Current arrival/completion classification within declared support.
    pub phase: GateActuatorPhase,
    /// Original current-target arrival certificate, absent without a finite all-position window.
    pub arrival: Option<GateActuatorArrivalCertificate>,
    /// Number of accepted numerically unequal target changes.
    pub target_change_count: usize,
    /// Number of activated non-hold episodes, including an initial active episode.
    pub episode_count: usize,
    /// Number of active episodes completing in accepted trials, without duplicate holds or proofs.
    pub arrival_count: usize,
    /// Current active episode identity, absent for an exact target hold.
    pub active_episode_index: Option<usize>,
    /// Latest raw causal packet, retaining original bits even for repeated numerical targets.
    pub latest_update: Option<GateTargetUpdate>,
    /// One bounded receipt for the latest actual target change.
    pub last_target_change: Option<GateActuatorTargetChangeReceipt>,
    /// One bounded proof for the latest completed active episode.
    pub last_completion: Option<GateActuatorCompletionReceipt>,
}

/// Immediate before/left/after state for one accepted candidate lane.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorStepReceipt {
    /// Index in the complete original structure context, never a compact subset index.
    pub element: usize,
    /// Exact actuator identity in immutable definition order.
    pub actuator_id: String,
    /// Complete immediate accepted state before this candidate interval.
    pub before: GateActuatorState,
    /// Complete old-target endpoint position before any target change.
    pub left_position: GateActuatorPosition,
    /// Old-target forward evidence before installing an endpoint target.
    pub left_forward: Option<GateActuatorForwardReceipt>,
    /// Original causal target packet, when supplied at this endpoint.
    pub update: Option<GateTargetUpdate>,
    /// Whether this accepted interval first completes an active episode.
    pub completed_episode: bool,
    /// Whether this endpoint installs a numerically unequal target.
    pub target_changed: bool,
    /// Complete immediate candidate state after old-target evaluation and target installation.
    pub after: GateActuatorState,
}

/// Original interval, endpoint event, and ordered complete lane receipts.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorStepDiagnostics {
    /// Original accepted interval start time in seconds.
    pub start_s: f64,
    /// Original candidate interval endpoint in seconds.
    pub end_s: f64,
    /// Exact endpoint event summary, absent for a substep ending before the next event.
    pub event: Option<GateActuatorEvent>,
    /// Ordered bounded receipts for the controlled gate subset.
    pub receipts: Vec<GateActuatorStepReceipt>,
}
