//! Candidate command continuation with ownership-based acceptance.

use super::{ControlCommandOutput, ControlState, ControlStepDiagnostics, OperatingControlBatch};
use crate::structures::StructureError;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Complete immutable candidate. Dropping it preserves the accepted batch.
#[derive(Debug)]
pub struct OperatingControlTrial<R: Runtime<DType = DType>> {
    pub(crate) next: OperatingControlBatch<R>,
}
impl<R: Runtime<DType = DType>> OperatingControlTrial<R> {
    /// Explicitly project right-interval candidate commands.
    pub fn commands(&self) -> Result<Vec<ControlCommandOutput>, StructureError> {
        self.next.commands()
    }
    /// Explicitly project complete candidate continuation.
    pub fn states(&self) -> Result<Vec<ControlState>, StructureError> {
        self.next.states()
    }
    /// Explicitly project both interval sides and simultaneous decision receipts.
    pub fn diagnostics(&self) -> Result<ControlStepDiagnostics, StructureError> {
        self.next.last_step()?.ok_or_else(|| {
            StructureError::numerical("control trial lacks its immediate step witness")
        })
    }
    /// Accept this complete candidate by ownership.
    pub fn into_next(self) -> OperatingControlBatch<R> {
        self.next
    }
}
