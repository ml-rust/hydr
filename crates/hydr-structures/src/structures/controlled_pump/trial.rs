//! Ownership-based acceptance of the complete sampled pump response continuation.

use super::{ControlledPumpBatch, ControlledPumpOutput};
use crate::structures::{OperatingControlBatch, StructureError, StructureOperation};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Immutable complete candidate. Discarding it preserves accepted control and response operation.
#[derive(Clone, Debug)]
pub struct ControlledPumpTrial<R: Runtime<DType = DType>> {
    pub(crate) next: ControlledPumpBatch<R>,
}
impl<R: Runtime<DType = DType>> ControlledPumpTrial<R> {
    /// Candidate controller memory and bounded decision receipts.
    pub fn controls(&self) -> &OperatingControlBatch<R> {
        self.next.controls()
    }
    /// Explicitly project candidate actual operation and response evidence.
    pub fn outputs(&self) -> Result<Vec<ControlledPumpOutput>, StructureError>
    where
        R::Client: numr::ops::ConditionalOps<R>,
    {
        self.next.outputs()
    }
    /// Explicitly project candidate original-order operation packets.
    pub fn operations(&self) -> Result<Vec<StructureOperation>, StructureError>
    where
        R::Client: numr::ops::ConditionalOps<R>,
    {
        self.next.operations()
    }
    /// Accept complete sampled response continuation by ownership, without numerical projection.
    pub fn into_next(self) -> ControlledPumpBatch<R> {
        self.next
    }
}
