//! Ownership-based acceptance of complete controller and actuator candidates.

use super::ControlledGateBatch;
use crate::structures::{GateActuatorBatch, OperatingControlBatch};

/// Complete immutable candidate pair. Discarding it preserves both accepted children.
#[derive(Clone, Debug)]
pub struct ControlledGateTrial {
    pub(crate) next: ControlledGateBatch,
}

impl ControlledGateTrial {
    /// Candidate commands and their complete bounded decision diagnostics.
    pub fn controls(&self) -> &OperatingControlBatch {
        self.next.controls()
    }

    /// Candidate actual opening and its complete bounded response diagnostics.
    pub fn actuators(&self) -> &GateActuatorBatch {
        self.next.actuators()
    }

    /// Accept both complete candidate children by ownership without state projection.
    pub fn into_next(self) -> ControlledGateBatch {
        self.next
    }
}
