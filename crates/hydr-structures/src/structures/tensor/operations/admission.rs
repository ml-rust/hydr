//! Original supplied operation identity, law, and physical range admission.
use numr::{dtype::DType, runtime::Runtime};

use super::TensorStructureContext;
use crate::structures::{
    StructureError, StructureInputErrorKind, StructureLaw, StructureOperatingPoint,
    StructureOperation,
};

pub(super) fn check<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    index: usize,
    operation: &StructureOperation,
) -> Result<bool, StructureError> {
    let element = &context.elements[index];
    if operation.structure_id != element.id {
        return Err(StructureError::input(
            StructureInputErrorKind::Identity,
            format!(
                "operation={:?},expected={:?}: match complete original structure order",
                operation.structure_id, element.id
            ),
        )
        .at(index));
    }
    let uninfluential = match (&element.law, operation.operating_point) {
        (StructureLaw::Pump(_), StructureOperatingPoint::PumpStoppedIsolated) => true,
        (StructureLaw::Pump(_), StructureOperatingPoint::PumpRunning { speed_ratio }) => {
            if !speed_ratio.is_finite() || speed_ratio <= 0.0 {
                return Err(StructureError::input(StructureInputErrorKind::OperatingPoint,format!("speed_ratio={speed_ratio}: supply positive finite speed or explicit stopped-isolated operation")).at(index));
            }
            false
        }
        (
            StructureLaw::HydrostaticRectangularGate(parameters),
            StructureOperatingPoint::GateOpening { opening_m },
        ) => {
            if !opening_m.is_finite()
                || opening_m < 0.0
                || opening_m > parameters.maximum_opening_m()
            {
                return Err(StructureError::input(
                    StructureInputErrorKind::OperatingPoint,
                    format!(
                "opening_m={opening_m},maximum_opening_m={}: supply a finite opening in the declared closed range",
                parameters.maximum_opening_m()
                    ),
                )
                .at(index));
            }
            opening_m == 0.0
        }
        (
            StructureLaw::Culvert(_)
            | StructureLaw::SubmergedOrifice(_)
            | StructureLaw::SuppressedRectangularWeir(_),
            StructureOperatingPoint::Passive,
        ) => false,
        (StructureLaw::Pump(_), _) => {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                "pump requires explicit running or stopped-isolated operation",
            )
            .at(index));
        }
        (StructureLaw::HydrostaticRectangularGate(_), _) => {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                "gate requires an explicit identity-bound opening operation",
            )
            .at(index));
        }
        _ => {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                "passive law requires an explicit Passive operation",
            )
            .at(index));
        }
    };
    Ok(uninfluential)
}
