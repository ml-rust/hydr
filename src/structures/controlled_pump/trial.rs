//! Ownership-based acceptance of the complete sampled pump response continuation.

use super::{ControlledPumpBatch, ControlledPumpOutput};
use crate::structures::{OperatingControlBatch, StructureError, StructureOperation};

/// Immutable complete candidate. Discarding it preserves accepted control and response operation.
#[derive(Clone, Debug)]
pub struct ControlledPumpTrial {
    pub(crate) next: ControlledPumpBatch,
}
impl ControlledPumpTrial {
    /// Candidate controller memory and bounded decision receipts.
    pub fn controls(&self) -> &OperatingControlBatch {
        self.next.controls()
    }
    /// Explicitly project candidate actual operation and response evidence.
    pub fn outputs(&self) -> Result<Vec<ControlledPumpOutput>, StructureError> {
        self.next.outputs()
    }
    /// Explicitly project candidate original-order operation packets.
    pub fn operations(&self) -> Result<Vec<StructureOperation>, StructureError> {
        self.next.operations()
    }
    /// Accept complete sampled response continuation by ownership, without numerical projection.
    pub fn into_next(self) -> ControlledPumpBatch {
        self.next
    }
}
