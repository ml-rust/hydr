//! Complete immutable original input compatibility without state projection.

use super::super::*;
use super::receipt_bits::ReceiptBits;
use crate::structures::{StructureError, StructureInputErrorKind};

pub(super) fn require(condition: bool, message: &str) -> Result<(), StructureError> {
    if condition {
        Ok(())
    } else {
        Err(StructureError::input(
            StructureInputErrorKind::Definition,
            message,
        ))
    }
}
pub(super) fn as_definition(error: StructureError) -> StructureError {
    match error {
        StructureError::Runtime { .. } => error,
        StructureError::InvalidInput {
            element, context, ..
        }
        | StructureError::Numerical { element, context } => StructureError::InvalidInput {
            element,
            kind: StructureInputErrorKind::Definition,
            context,
        },
    }
}
pub(super) fn check(
    batch: &GateActuatorBatch,
    data: &GateActuatorSnapshotData,
) -> Result<(), StructureError> {
    require(
        crate::structures::identity::same_elements(batch.elements(), &data.elements)
            && batch.definitions().same_bits(data.definitions.as_slice())
            && batch
                .initial_states()
                .same_bits(data.initial_states.as_slice())
            && batch.schedule().same_bits(&data.schedule),
        "actuator snapshot original context differs: restore into the identically constructed actual gate batch",
    )
}
pub(super) fn trusted(a: &GateActuatorBatch, b: &GateActuatorBatch) -> Result<(), StructureError> {
    if std::sync::Arc::ptr_eq(&a.context, &b.context) {
        return Ok(());
    }
    require(
        crate::structures::identity::same_elements(a.elements(), b.elements())
            && a.definitions().same_bits(b.definitions())
            && a.initial_states().same_bits(b.initial_states())
            && a.schedule().same_bits(b.schedule()),
        "trusted actuator snapshot differs from destination original context",
    )
}
