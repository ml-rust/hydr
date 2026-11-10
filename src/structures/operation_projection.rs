//! Deferred original operation sources and explicit complete packet projection.

use super::{
    GateTrajectoryEvaluationData, StructureError, StructureLaw, StructureOperatingPoint,
    StructureOperation,
};
use std::sync::Arc;

#[derive(Debug)]
pub(super) enum ProposalOperations {
    Supplied(Vec<StructureOperation>),
    Trajectory {
        evaluation: Arc<GateTrajectoryEvaluationData>,
        other: Vec<StructureOperation>,
    },
}
pub(super) fn project(
    source: &ProposalOperations,
) -> Result<Vec<StructureOperation>, StructureError> {
    match source {
        ProposalOperations::Supplied(operations) => Ok(operations.clone()),
        ProposalOperations::Trajectory { evaluation, other } => {
            let context = &evaluation.binding.context;
            let opening = super::projection::read(&evaluation.opening, context.elements.len())?;
            let mut supplied = other.iter();
            context
                .elements
                .iter()
                .enumerate()
                .map(|(index, element)| {
                    if matches!(element.law, StructureLaw::HydrostaticRectangularGate(_)) {
                        Ok(StructureOperation {
                            structure_id: element.id.clone(),
                            operating_point: StructureOperatingPoint::GateOpening {
                                opening_m: opening[index],
                            },
                        })
                    } else {
                        supplied.next().cloned().ok_or_else(|| {
                            StructureError::numerical(
                                "checked non-gate operation source is incomplete",
                            )
                            .at(index)
                        })
                    }
                })
                .collect()
        }
    }
}
