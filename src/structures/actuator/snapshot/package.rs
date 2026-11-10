//! Trusted tensor clones and explicit complete caller-owned projection.

use super::super::*;
use crate::structures::StructureError;

/// Complete trusted continuation without state projection or recursive history.
#[derive(Clone, Debug)]
pub struct GateActuatorSnapshot {
    pub(super) batch: GateActuatorBatch,
}
impl GateActuatorSnapshot {
    /// Accepted snapshot time in seconds.
    pub fn time(&self) -> f64 {
        self.batch.time()
    }
    /// Explicitly project the complete caller-owned continuation package.
    pub fn data(&self) -> Result<GateActuatorSnapshotData, StructureError> {
        self.batch.snapshot_data()
    }
}
impl GateActuatorBatch {
    /// Clone complete trusted continuation without derived array readback.
    pub fn snapshot(&self) -> GateActuatorSnapshot {
        GateActuatorSnapshot {
            batch: self.clone(),
        }
    }
    /// Explicitly project complete original inputs and bounded accepted receipts.
    pub fn snapshot_data(&self) -> Result<GateActuatorSnapshotData, StructureError> {
        Ok(GateActuatorSnapshotData {
            elements: self.elements().to_vec(),
            definitions: self.definitions().to_vec(),
            initial_states: self.initial_states().to_vec(),
            schedule: self.schedule().clone(),
            time_s: self.time,
            next_target_index: self.next_target,
            accepted_steps: self.steps,
            states: self.states()?,
            last_step: self.last_step()?,
        })
    }
}
