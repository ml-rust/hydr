//! Explicit finite-rate actuator package contracts.

use super::*;
use crate::structures::StructureElement;
/// Caller-owned complete continuation. Encoding must preserve every original float bit.
#[derive(Clone, Debug, PartialEq)]
pub struct GateActuatorSnapshotData {
    /// Complete immutable original structure definitions, including uncontrolled elements.
    pub elements: Vec<StructureElement>,
    /// Ordered immutable actuator definitions and explicit numerical policies.
    pub definitions: Vec<GateActuatorDefinition>,
    /// Original supplied initial actual openings and target bits.
    pub initial_states: Vec<GateActuatorInitialState>,
    /// Original immutable target calendar and terminal support.
    pub schedule: GateActuatorSchedule,
    /// Original accepted or retained event time in seconds.
    pub time_s: f64,
    /// Index of the next unprocessed original target event.
    pub next_target_index: usize,
    /// Number of accepted positive-duration intervals.
    pub accepted_steps: usize,
    /// Complete current ordered actuator continuation values.
    pub states: Vec<GateActuatorState>,
    /// Bounded immediate final interval evidence, absent for zero accepted history.
    pub last_step: Option<GateActuatorStepDiagnostics>,
}
