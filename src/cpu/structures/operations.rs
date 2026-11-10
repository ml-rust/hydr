//! Identity-bound operations and opaque prescribed gate descriptors.

use super::CpuStructureContext;
use crate::cpu::CpuTensor;
use crate::structures::{
    GateTrajectoryEvaluationData, StructureError, StructureInputErrorKind, StructureLaw,
    StructureOperatingPoint, StructureOperation,
};

pub(crate) enum CheckedOperationSlot<'a> {
    Supplied(&'a StructureOperation),
    TrajectoryGate,
}
pub(crate) struct CheckedOperations<'a> {
    slots: Vec<CheckedOperationSlot<'a>>,
    isolated: Vec<bool>,
    opening: Option<&'a CpuTensor>,
}
impl<'a> CheckedOperations<'a> {
    pub(crate) fn new(
        context: &CpuStructureContext,
        packets: &'a [StructureOperation],
    ) -> Result<Self, StructureError> {
        if packets.len() != context.elements.len() {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                format!(
                    "operation count={},structure count={}: supply complete ordered operations",
                    packets.len(),
                    context.elements.len()
                ),
            ));
        }
        let mut isolated = Vec::with_capacity(packets.len());
        for (index, operation) in packets.iter().enumerate() {
            isolated.push(check(context, index, operation)?);
        }
        Ok(Self {
            slots: packets.iter().map(CheckedOperationSlot::Supplied).collect(),
            isolated,
            opening: None,
        })
    }
    pub(crate) fn trajectory(
        context: &CpuStructureContext,
        evaluation: &'a GateTrajectoryEvaluationData,
        other: &'a [StructureOperation],
    ) -> Result<Self, StructureError> {
        let expected = context.elements.len() - evaluation.binding.slots.len();
        if other.len() != expected {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                format!(
                    "non-gate operation count={},expected={expected}: supply every non-gate operation in original subsequence order",
                    other.len()
                ),
            ));
        }
        let mut slots = Vec::with_capacity(context.elements.len());
        let mut isolated = Vec::with_capacity(context.elements.len());
        let (mut gate, mut supplied) = (0, 0);
        for (index, element) in context.elements.iter().enumerate() {
            if matches!(element.law, StructureLaw::HydrostaticRectangularGate(_)) {
                if evaluation.binding.slots.get(gate) != Some(&index) {
                    return Err(StructureError::numerical(
                        "trajectory binding does not match complete original context",
                    )
                    .at(index));
                }
                slots.push(CheckedOperationSlot::TrajectoryGate);
                isolated.push(evaluation.metadata[gate].closed_by_definition);
                gate += 1;
            } else {
                let operation = &other[supplied];
                if matches!(
                    operation.operating_point,
                    StructureOperatingPoint::GateOpening { .. }
                ) {
                    return Err(StructureError::input(StructureInputErrorKind::Identity,"non-gate operation overrides prescribed gate: supply original non-gate operations").at(index));
                }
                isolated.push(check(context, index, operation)?);
                slots.push(CheckedOperationSlot::Supplied(operation));
                supplied += 1;
            }
        }
        Ok(Self {
            slots,
            isolated,
            opening: Some(&evaluation.opening),
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }
    pub(crate) fn supplied(&self, index: usize) -> Option<StructureOperatingPoint> {
        match &self.slots[index] {
            CheckedOperationSlot::Supplied(packet) => Some(packet.operating_point),
            CheckedOperationSlot::TrajectoryGate => None,
        }
    }
    pub(crate) fn opening(&self) -> Option<&CpuTensor> {
        self.opening
    }
    pub(crate) fn isolated(&self, index: usize) -> bool {
        self.isolated[index]
    }
}
fn check(
    context: &CpuStructureContext,
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
            StructureLaw::SubmergedOrifice(_) | StructureLaw::SuppressedRectangularWeir(_),
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
