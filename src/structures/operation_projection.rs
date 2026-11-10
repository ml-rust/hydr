//! Deferred original operation sources and explicit complete packet projection.

use super::{
    GateTrajectoryEvaluationData, StructureError, StructureOperatingPoint, StructureOperation,
};
use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use std::sync::Arc;

#[derive(Debug)]
pub(super) enum ProposalOperations {
    Supplied(Vec<StructureOperation>),
    Trajectory {
        evaluation: Arc<GateTrajectoryEvaluationData>,
        other: Vec<StructureOperation>,
    },
    Derived {
        context: Arc<CpuStructureContext>,
        slots: Vec<usize>,
        opening: CpuTensor,
        manual: Vec<StructureOperation>,
    },
}
pub(super) fn project(
    source: &ProposalOperations,
) -> Result<Vec<StructureOperation>, StructureError> {
    match source {
        ProposalOperations::Supplied(operations) => Ok(operations.clone()),
        ProposalOperations::Trajectory { evaluation, other } => project_derived(
            &evaluation.binding.context,
            &evaluation.binding.slots,
            &evaluation.opening,
            other,
        ),
        ProposalOperations::Derived {
            context,
            slots,
            opening,
            manual,
        } => project_derived(context, slots, opening, manual),
    }
}
fn project_derived(
    context: &CpuStructureContext,
    slots: &[usize],
    opening: &CpuTensor,
    manual: &[StructureOperation],
) -> Result<Vec<StructureOperation>, StructureError> {
    let opening = super::projection::read(opening, context.elements.len())?;
    let mut derived = slots.iter().copied().peekable();
    let mut supplied = manual.iter();
    let operations = context
        .elements
        .iter()
        .enumerate()
        .map(|(index, element)| {
            if derived.peek() == Some(&index) {
                derived.next();
                Ok(StructureOperation {
                    structure_id: element.id.clone(),
                    operating_point: StructureOperatingPoint::GateOpening {
                        opening_m: opening[index],
                    },
                })
            } else {
                supplied.next().cloned().ok_or_else(|| {
                    StructureError::numerical("checked manual operation complement is incomplete")
                        .at(index)
                })
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    if derived.next().is_some() || supplied.next().is_some() {
        return Err(StructureError::numerical(
            "checked operation projection has extra or unordered source slots",
        ));
    }
    Ok(operations)
}
