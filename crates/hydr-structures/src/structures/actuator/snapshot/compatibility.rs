//! Complete immutable original input compatibility without state projection.

use super::super::*;
use crate::structures::{StructureError, StructureInputErrorKind};
use numr::dtype::DType;
use numr::runtime::Runtime;

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
pub(super) fn check<R: Runtime<DType = DType>>(
    batch: &GateActuatorBatch<R>,
    data: &GateActuatorSnapshotData,
) -> Result<(), StructureError> {
    require(
        super::super::compatibility::same_original(
            &batch.context,
            &data.elements,
            &data.definitions,
            &data.initial_states,
            &data.schedule,
        ),
        "actuator snapshot original context differs: restore into the identically constructed actual gate batch",
    )
}

pub(super) fn trusted<R: Runtime<DType = DType>>(
    a: &GateActuatorBatch<R>,
    b: &GateActuatorBatch<R>,
) -> Result<(), StructureError> {
    require(
        super::super::compatibility::same_context(&a.context, &b.context),
        "trusted actuator snapshot differs from destination original context",
    )
}
