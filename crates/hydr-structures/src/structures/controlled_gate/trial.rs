//! Ownership-based acceptance of complete controller and actuator candidates.

use super::ControlledGateBatch;
use crate::structures::{GateActuatorBatch, OperatingControlBatch};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Complete immutable candidate pair. Discarding it preserves both accepted children.
#[derive(Clone, Debug)]
pub struct ControlledGateTrial<R: Runtime<DType = DType>> {
    pub(crate) next: ControlledGateBatch<R>,
}

impl<R: Runtime<DType = DType>> ControlledGateTrial<R> {
    /// Candidate commands and their complete bounded decision diagnostics.
    pub fn controls(&self) -> &OperatingControlBatch<R> {
        self.next.controls()
    }

    /// Candidate actual opening and its complete bounded response diagnostics.
    pub fn actuators(&self) -> &GateActuatorBatch<R> {
        self.next.actuators()
    }

    /// Accept both complete candidate children by ownership without state projection.
    pub fn into_next(self) -> ControlledGateBatch<R> {
        self.next
    }
}
