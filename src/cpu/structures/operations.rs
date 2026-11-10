//! Identity-bound manual packets and typed derived operation descriptors.

mod admission;
mod derived;

use super::CpuStructureContext;
use crate::cpu::CpuTensor;
use crate::structures::{
    GateTrajectoryEvaluationData, StructureError, StructureInputErrorKind, StructureLaw,
    StructureOperatingPoint, StructureOperation,
};
use numr::ops::ConditionalOps;

use admission::check;

/// Checked registry tensor inputs in complete original structure layout.
pub(crate) struct DerivedGateOperations<'a> {
    pub slots: &'a [usize],
    pub mask: &'a CpuTensor,
    pub opening: &'a CpuTensor,
    pub isolated: &'a CpuTensor,
}
/// Checked selected pump operation tensors in complete original structure layout.
#[derive(Clone, Copy)]
pub(crate) struct DerivedPumpOperations<'a> {
    pub slots: &'a [usize],
    pub mask: &'a CpuTensor,
    pub running: &'a CpuTensor,
    pub isolated: &'a CpuTensor,
    pub speed: &'a CpuTensor,
    pub power_of_two: &'a CpuTensor,
}
/// Retained selected operations for explicit proposal projection.
#[derive(Debug)]
pub(crate) struct DerivedPumpOperationData {
    pub slots: Vec<usize>,
    pub running: CpuTensor,
    pub speed: CpuTensor,
}
struct GateBinding<'a> {
    slots: &'a [usize],
    mask: &'a CpuTensor,
    opening: &'a CpuTensor,
    isolated: CpuTensor,
}
pub(crate) enum CheckedOperationSlot<'a> {
    Supplied(&'a StructureOperation),
    DerivedGate,
    DerivedPump,
}
pub(crate) struct CheckedOperations<'a> {
    slots: Vec<CheckedOperationSlot<'a>>,
    isolated: CpuTensor,
    derived_mask: Option<&'a CpuTensor>,
    opening: Option<&'a CpuTensor>,
    pump: Option<DerivedPumpOperations<'a>>,
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
        let isolated = packets
            .iter()
            .enumerate()
            .map(|(index, operation)| {
                check(context, index, operation).map(|value| if value { 1.0 } else { 0.0 })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            slots: packets.iter().map(CheckedOperationSlot::Supplied).collect(),
            isolated: context.tensor(&isolated)?,
            derived_mask: None,
            opening: None,
            pump: None,
        })
    }
    pub(crate) fn trajectory(
        context: &CpuStructureContext,
        evaluation: &'a GateTrajectoryEvaluationData,
        other: &'a [StructureOperation],
    ) -> Result<Self, StructureError> {
        let mut closed = vec![0.0; context.elements.len()];
        if evaluation.metadata.len() != evaluation.binding.slots.len() {
            return Err(StructureError::numerical(
                "trajectory metadata does not match its assigned gate subset",
            ));
        }
        for (&slot, metadata) in evaluation.binding.slots.iter().zip(&evaluation.metadata) {
            if let Some(value) = closed.get_mut(slot) {
                *value = if metadata.closed_by_definition {
                    1.0
                } else {
                    0.0
                };
            } else {
                return Err(StructureError::numerical(
                    "trajectory binding exceeds its complete actual context",
                ));
            }
        }
        Self::bind(
            context,
            other,
            Some(GateBinding {
                slots: &evaluation.binding.slots,
                mask: &evaluation.binding.tensors.assigned_mask,
                opening: &evaluation.opening,
                isolated: context.tensor(&closed)?,
            }),
            None,
            true,
        )
    }
    pub(crate) fn derived(
        context: &CpuStructureContext,
        packets: &'a [StructureOperation],
        gates: Option<DerivedGateOperations<'a>>,
        pumps: Option<DerivedPumpOperations<'a>>,
    ) -> Result<Self, StructureError> {
        Self::bind(
            context,
            packets,
            gates.map(|gate| GateBinding {
                slots: gate.slots,
                mask: gate.mask,
                opening: gate.opening,
                isolated: gate.isolated.clone(),
            }),
            pumps,
            false,
        )
    }
    fn bind(
        context: &CpuStructureContext,
        packets: &'a [StructureOperation],
        gates: Option<GateBinding<'a>>,
        pumps: Option<DerivedPumpOperations<'a>>,
        trajectory: bool,
    ) -> Result<Self, StructureError> {
        let n = context.elements.len();
        let gate_flags = gates
            .as_ref()
            .map(|gate| derived::gate_slots(context, gate.slots))
            .transpose()?;
        let pump_flags = pumps
            .as_ref()
            .map(|pump| derived::pump_slots(context, pump.slots))
            .transpose()?;
        let mut flags = vec![0u8; n];
        for index in 0..n {
            let gate = gate_flags
                .as_ref()
                .is_some_and(|values| values[index] == 1.0);
            let pump = pump_flags
                .as_ref()
                .is_some_and(|values| values[index] == 1.0);
            if gate && pump {
                return Err(StructureError::input(
                    StructureInputErrorKind::Identity,
                    "derived operation ownership overlaps: assign one typed operation per actual structure",
                ).at(index));
            }
            flags[index] = if gate {
                1
            } else if pump {
                2
            } else {
                0
            };
        }
        let expected = flags.iter().filter(|&&assigned| assigned == 0).count();
        if packets.len() != expected {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                format!(
                    "manual operation count={},expected={expected}: supply complete complement operations in original subsequence order",
                    packets.len()
                ),
            ));
        }
        if let (Some(gate), Some(flags)) = (&gates, &gate_flags) {
            derived::gate(context, gate, flags)?;
        }
        if let (Some(pump), Some(flags)) = (&pumps, &pump_flags) {
            derived::pump(context, pump, flags)?;
        }
        let c = &context.client;
        let mut slots = Vec::with_capacity(n);
        let mut supplied = packets.iter();
        let mut isolated = vec![0.0; n];
        for (index, &assigned) in flags.iter().enumerate() {
            if assigned == 1 {
                slots.push(CheckedOperationSlot::DerivedGate);
            } else if assigned == 2 {
                slots.push(CheckedOperationSlot::DerivedPump);
            } else {
                let operation = supplied.next().ok_or_else(|| {
                    StructureError::numerical("checked manual complement is incomplete").at(index)
                })?;
                if trajectory
                    && !matches!(
                        context.elements[index].law,
                        StructureLaw::HydrostaticRectangularGate(_)
                    )
                    && matches!(
                        operation.operating_point,
                        StructureOperatingPoint::GateOpening { .. }
                    )
                {
                    return Err(StructureError::input(StructureInputErrorKind::Identity,"non-gate operation overrides prescribed gate: supply original complement operations").at(index));
                }
                isolated[index] = if check(context, index, operation)? {
                    1.0
                } else {
                    0.0
                };
                slots.push(CheckedOperationSlot::Supplied(operation));
            }
        }
        let mut isolated = context.tensor(&isolated)?;
        if let Some(gate) = &gates {
            isolated = c.where_cond(gate.mask, &gate.isolated, &isolated)?;
        }
        if let Some(pump) = &pumps {
            isolated = c.where_cond(pump.mask, pump.isolated, &isolated)?;
        }
        Ok(Self {
            slots,
            isolated,
            derived_mask: gates.as_ref().map(|gate| gate.mask),
            opening: gates.as_ref().map(|gate| gate.opening),
            pump: pumps,
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }
    pub(crate) fn supplied(&self, index: usize) -> Option<StructureOperatingPoint> {
        match &self.slots[index] {
            CheckedOperationSlot::Supplied(packet) => Some(packet.operating_point),
            CheckedOperationSlot::DerivedGate | CheckedOperationSlot::DerivedPump => None,
        }
    }
    pub(crate) fn derived_opening(&self) -> Option<(&CpuTensor, &CpuTensor)> {
        self.opening.zip(self.derived_mask)
    }
    pub(crate) fn isolated(&self) -> &CpuTensor {
        &self.isolated
    }
    pub(crate) fn derived_pump(&self) -> Option<&DerivedPumpOperations<'a>> {
        self.pump.as_ref()
    }
}
