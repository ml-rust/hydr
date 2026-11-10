//! Deferred original operation sources and explicit complete packet projection.
use numr::{dtype::DType, runtime::Runtime};

use super::{
    GateTrajectoryEvaluationData, StructureError, StructureOperatingPoint, StructureOperation,
};
use crate::tensor::structures::{DerivedPumpOperationData, TensorStructureContext};
use numr::tensor::Tensor;
use std::sync::Arc;

#[derive(Debug)]
pub(super) enum ProposalOperations<R: Runtime<DType = DType>> {
    Supplied(Vec<StructureOperation>),
    Trajectory {
        evaluation: Arc<GateTrajectoryEvaluationData<R>>,
        other: Vec<StructureOperation>,
    },
    Derived {
        context: Arc<TensorStructureContext<R>>,
        slots: Vec<usize>,
        opening: Tensor<R>,
        pumps: Option<Box<DerivedPumpOperationData<R>>>,
        manual: Vec<StructureOperation>,
    },
}
pub(super) fn project<R: Runtime<DType = DType>>(
    source: &ProposalOperations<R>,
) -> Result<Vec<StructureOperation>, StructureError> {
    match source {
        ProposalOperations::Supplied(operations) => Ok(operations.clone()),
        ProposalOperations::Trajectory { evaluation, other } => project_derived(
            &evaluation.binding.context,
            &evaluation.binding.slots,
            &evaluation.opening,
            other,
            None,
        ),
        ProposalOperations::Derived {
            context,
            slots,
            opening,
            pumps,
            manual,
        } => project_derived(context, slots, opening, manual, pumps.as_deref()),
    }
}
fn project_derived<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    slots: &[usize],
    opening: &Tensor<R>,
    manual: &[StructureOperation],
    pumps: Option<&DerivedPumpOperationData<R>>,
) -> Result<Vec<StructureOperation>, StructureError> {
    let opening = super::projection::read(opening, context.elements.len())?;
    let pump_values = pumps
        .map(|pump| {
            Ok::<_, StructureError>((
                super::projection::read(&pump.running, context.elements.len())?,
                super::projection::read(&pump.speed, context.elements.len())?,
            ))
        })
        .transpose()?;
    let mut derived = slots.iter().copied().peekable();
    let mut pump_slots = pumps
        .map_or(&[][..], |pump| pump.slots.as_slice())
        .iter()
        .copied()
        .peekable();
    let mut supplied = manual.iter();
    let operations = context
        .elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            let gate = derived.peek() == Some(&index);
            let pump = pump_slots.peek() == Some(&index);
            if gate && pump {
                return Err(StructureError::numerical(
                    "checked operation projection contains overlapping typed ownership",
                )
                .at(index));
            }
            if gate {
                derived.next();
                Ok(StructureOperation {
                    structure_id: element.id.clone(),
                    operating_point: StructureOperatingPoint::GateOpening {
                        opening_m: opening[index],
                    },
                })
            } else if pump {
                pump_slots.next();
                let (running, speed) = pump_values.as_ref().ok_or_else(|| {
                    StructureError::numerical(
                        "checked pump projection lacks selected operation tensors",
                    )
                    .at(index)
                })?;
                let operating_point = if running[index] == 1.0 {
                    StructureOperatingPoint::PumpRunning {
                        speed_ratio: speed[index],
                    }
                } else if running[index] == 0.0 {
                    StructureOperatingPoint::PumpStoppedIsolated
                } else {
                    return Err(StructureError::numerical(
                        "checked pump projection loses its binary operation identity",
                    )
                    .at(index));
                };
                Ok(StructureOperation {
                    structure_id: element.id.clone(),
                    operating_point,
                })
            } else {
                supplied.next().cloned().ok_or_else(|| {
                    StructureError::numerical("checked manual operation complement is incomplete")
                        .at(index)
                })
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if derived.next().is_some() || pump_slots.next().is_some() || supplied.next().is_some() {
        return Err(StructureError::numerical(
            "checked operation projection has extra or unordered source slots",
        ));
    }
    Ok(operations)
}
