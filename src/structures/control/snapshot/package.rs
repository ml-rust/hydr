//! Complete caller-owned continuation and trusted tensor snapshots.

use super::super::{
    ControlDecisionSchedule, ControlInitialState, ControlState, ControlStepDiagnostics,
    OperatingControlBatch, OperatingControlDefinition, SampledControlTiming,
};
use crate::structures::{StructureElement, StructureError};

/// Complete caller-owned control continuation. Preserve every original F64 bit when encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatingControlSnapshotData {
    /// Complete immutable actual structure definitions.
    pub elements: Vec<StructureElement>,
    /// Original ordered controller definitions.
    pub definitions: Vec<OperatingControlDefinition>,
    /// Complete assumed initial memory and antecedents.
    pub initial_states: Vec<ControlInitialState>,
    /// Complete original future decision schedule.
    pub schedule: ControlDecisionSchedule,
    /// Original sampled applicability and timing policy.
    pub timing: SampledControlTiming,
    /// Accepted time in seconds.
    pub time_s: f64,
    /// Index of the first unprocessed original decision.
    pub next_decision_index: usize,
    /// Accepted original interval count.
    pub accepted_steps: usize,
    /// Complete current controller continuation.
    pub states: Vec<ControlState>,
    /// Complete last accepted interval, absent before any accepted interval.
    pub last_step: Option<ControlStepDiagnostics>,
}
/// Trusted complete tensor clone without explicit projection or history chains.
#[derive(Clone, Debug)]
pub struct OperatingControlSnapshot {
    pub(super) batch: OperatingControlBatch,
}
impl OperatingControlSnapshot {
    /// Accepted snapshot time in seconds.
    pub fn time(&self) -> f64 {
        self.batch.time()
    }
    /// Explicitly project complete caller-owned values.
    pub fn data(&self) -> Result<OperatingControlSnapshotData, StructureError> {
        self.batch.snapshot_data()
    }
}
impl OperatingControlBatch {
    /// Clone complete trusted continuation without state projection.
    pub fn snapshot(&self) -> OperatingControlSnapshot {
        OperatingControlSnapshot {
            batch: self.clone(),
        }
    }
    /// Explicitly project every value required for equivalent future continuation.
    pub fn snapshot_data(&self) -> Result<OperatingControlSnapshotData, StructureError> {
        Ok(OperatingControlSnapshotData {
            elements: self.elements().to_vec(),
            definitions: self.definitions().to_vec(),
            initial_states: self.initial_states().to_vec(),
            schedule: self.schedule().clone(),
            timing: self.timing().clone(),
            time_s: self.time,
            next_decision_index: self.next_decision,
            accepted_steps: self.steps,
            states: self.states()?,
            last_step: self.last_step()?,
        })
    }
}
