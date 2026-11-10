//! Ownership-based acceptance of complete candidate actuator state.

use super::{
    GateActuatorBatch, GateActuatorOutput, GateActuatorState, GateActuatorStepDiagnostics,
};
use crate::structures::StructureError;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Immutable complete candidate. Dropping it preserves every accepted value.
#[derive(Clone, Debug)]
pub struct GateActuatorTrial<R: Runtime<DType = DType>> {
    pub(crate) next: GateActuatorBatch<R>,
}
impl<R: Runtime<DType = DType>> GateActuatorTrial<R> {
    /// Explicitly project candidate actual opening, support, and effective arrival bounds.
    pub fn outputs(&self) -> Result<Vec<GateActuatorOutput>, StructureError> {
        self.next.outputs()
    }
    /// Explicitly project complete candidate continuation values.
    pub fn states(&self) -> Result<Vec<GateActuatorState>, StructureError> {
        self.next.states()
    }
    /// Explicitly project the candidate interval and bounded original-input receipts.
    pub fn diagnostics(&self) -> Result<GateActuatorStepDiagnostics, StructureError> {
        self.next.last_step()?.ok_or_else(|| {
            StructureError::numerical("actuator trial lacks its immediate step receipt")
        })
    }
    /// Accept the complete candidate by ownership without projecting numerical state.
    pub fn into_next(self) -> GateActuatorBatch<R> {
        self.next
    }
}
