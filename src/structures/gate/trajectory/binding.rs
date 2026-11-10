//! Original gate-subsequence assignments and immutable context ownership.
use numr::{dtype::DType, runtime::Runtime};

use super::{GateOpeningTrajectory, GateTrajectoryEvent, GateTrajectorySupport};
use crate::structures::{StructureError, StructureInputErrorKind, StructureLaw};
use crate::tensor::structures::{
    TensorStructureContext, gate::trajectory::TensorTrajectoryContext,
};
use std::collections::BTreeMap;
use std::sync::Arc;

/// One exact original gate identity and its complete prescribed definition.
#[derive(Clone, Debug, PartialEq)]
pub struct GateTrajectoryAssignment {
    /// Identity in the original gate subsequence.
    pub structure_id: String,
    /// Complete immutable operating trajectory.
    pub trajectory: GateOpeningTrajectory,
}
#[derive(Debug)]
pub(crate) struct GateTrajectoryData<R: Runtime<DType = DType>> {
    pub context: Arc<TensorStructureContext<R>>,
    pub assignments: Vec<GateTrajectoryAssignment>,
    pub slots: Vec<usize>,
    pub support: GateTrajectorySupport,
    pub events: Vec<GateTrajectoryEvent>,
    pub tensors: TensorTrajectoryContext<R>,
}
impl<R: Runtime<DType = DType>> GateTrajectoryData<R> {
    pub(super) fn new(
        context: Arc<TensorStructureContext<R>>,
        assignments: Vec<GateTrajectoryAssignment>,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let slots: Vec<_> = context
            .elements
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                matches!(e.law, StructureLaw::HydrostaticRectangularGate(_)).then_some(i)
            })
            .collect();
        if slots.is_empty() {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                "gate trajectory batch has no gates: supply an actual gate law",
            ));
        }
        if assignments.len() != slots.len() {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                format!(
                    "gate assignments={},expected={}: supply every gate in original subsequence order",
                    assignments.len(),
                    slots.len()
                ),
            ));
        }
        for (assignment, &slot) in assignments.iter().zip(&slots) {
            let element = &context.elements[slot];
            if assignment.structure_id != element.id {
                return Err(StructureError::input(
                    StructureInputErrorKind::Identity,
                    format!(
                        "gate assignment={:?},expected={:?}: match original gate subsequence",
                        assignment.structure_id, element.id
                    ),
                )
                .at(slot));
            }
            check_range(&context, assignment, slot)?;
        }
        Self::build(context, assignments, slots, false)
    }
    pub(super) fn subset(
        context: Arc<TensorStructureContext<R>>,
        assignments: Vec<GateTrajectoryAssignment>,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        if assignments.is_empty() {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                "gate trajectory subset is empty: supply an explicit nonempty ordered gate subset",
            ));
        }
        let index: BTreeMap<_, _> = context
            .elements
            .iter()
            .enumerate()
            .map(|(slot, element)| (element.id.as_str(), slot))
            .collect();
        let mut slots = Vec::with_capacity(assignments.len());
        for assignment in &assignments {
            let slot = index.get(assignment.structure_id.as_str()).copied().ok_or_else(|| {
                StructureError::input(
                    StructureInputErrorKind::Identity,
                    format!(
                        "gate assignment={:?}: supply an identity from the complete original structure context",
                        assignment.structure_id
                    ),
                )
            })?;
            if !matches!(
                context.elements[slot].law,
                StructureLaw::HydrostaticRectangularGate(_)
            ) {
                return Err(StructureError::input(
                    StructureInputErrorKind::Identity,
                    "gate trajectory assignment requires an actual hydrostatic gate",
                )
                .at(slot));
            }
            if slots.last().is_some_and(|previous| *previous >= slot) {
                return Err(StructureError::input(
                    StructureInputErrorKind::Identity,
                    "gate trajectory subset must be unique and follow original structure order",
                )
                .at(slot));
            }
            check_range(&context, assignment, slot)?;
            slots.push(slot);
        }
        Self::build(context, assignments, slots, true)
    }
    fn build(
        context: Arc<TensorStructureContext<R>>,
        assignments: Vec<GateTrajectoryAssignment>,
        slots: Vec<usize>,
        allow_point: bool,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let (support, events) = super::events::build(&assignments, allow_point)?;
        let tensors = TensorTrajectoryContext::new(&context, &assignments, &slots)?;
        Ok(Self {
            context,
            assignments,
            slots,
            support,
            events,
            tensors,
        })
    }
}

fn check_range<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    assignment: &GateTrajectoryAssignment,
    slot: usize,
) -> Result<(), StructureError> {
    if let StructureLaw::HydrostaticRectangularGate(parameters) = &context.elements[slot].law {
        for (knot_index, knot) in assignment.trajectory.knots().iter().enumerate() {
            if knot.opening_m > parameters.maximum_opening_m() {
                return Err(StructureError::input(StructureInputErrorKind::Definition,format!("gate knot={knot_index},opening_m={},maximum={}: supply opening within actual gate range",knot.opening_m,parameters.maximum_opening_m())).at(slot));
            }
        }
    }
    Ok(())
}
