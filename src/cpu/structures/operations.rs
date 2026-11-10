//! Identity-bound manual packets and static derived-gate tensor descriptors.

use super::CpuStructureContext;
use crate::cpu::CpuTensor;
use crate::structures::{
    GateTrajectoryEvaluationData, StructureError, StructureInputErrorKind, StructureLaw,
    StructureOperatingPoint, StructureOperation,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Device;

/// Checked registry tensor inputs in complete original structure layout.
pub(crate) struct DerivedGateOperations<'a> {
    pub slots: &'a [usize],
    pub mask: &'a CpuTensor,
    pub opening: &'a CpuTensor,
    pub isolated: &'a CpuTensor,
}
pub(crate) enum CheckedOperationSlot<'a> {
    Supplied(&'a StructureOperation),
    DerivedGate,
}
pub(crate) struct CheckedOperations<'a> {
    slots: Vec<CheckedOperationSlot<'a>>,
    isolated: CpuTensor,
    derived_mask: Option<&'a CpuTensor>,
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
            &evaluation.binding.slots,
            &evaluation.binding.tensors.assigned_mask,
            &evaluation.opening,
            context.tensor(&closed)?,
            true,
        )
    }
    pub(crate) fn derived(
        context: &CpuStructureContext,
        packets: &'a [StructureOperation],
        derived: DerivedGateOperations<'a>,
    ) -> Result<Self, StructureError> {
        Self::bind(
            context,
            packets,
            derived.slots,
            derived.mask,
            derived.opening,
            derived.isolated.clone(),
            false,
        )
    }
    fn bind(
        context: &CpuStructureContext,
        packets: &'a [StructureOperation],
        derived_slots: &[usize],
        mask: &'a CpuTensor,
        opening: &'a CpuTensor,
        derived_isolated: CpuTensor,
        trajectory: bool,
    ) -> Result<Self, StructureError> {
        let n = context.elements.len();
        let mut flags = vec![0.0; n];
        let mut previous = None;
        for &slot in derived_slots {
            if slot >= n || previous.is_some_and(|old| old >= slot) {
                return Err(StructureError::input(
                    StructureInputErrorKind::Identity,
                    "derived gates must follow unique complete-context order",
                ));
            }
            if !matches!(
                context.elements[slot].law,
                StructureLaw::HydrostaticRectangularGate(_)
            ) {
                return Err(StructureError::input(
                    StructureInputErrorKind::Identity,
                    "derived operation ownership requires an actual gate",
                )
                .at(slot));
            }
            flags[slot] = 1.0;
            previous = Some(slot);
        }
        let expected = n - derived_slots.len();
        if packets.len() != expected {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                format!(
                    "manual operation count={},expected={expected}: supply complete complement operations in original subsequence order",
                    packets.len()
                ),
            ));
        }
        for value in [mask, opening, &derived_isolated] {
            if value.shape() != [n]
                || value.dtype() != DType::F64
                || value.device().id() != context.device.id()
            {
                return Err(StructureError::numerical(
                    "derived gate tensor metadata differs from complete CPU F64 structure layout",
                ));
            }
        }
        let c = &context.client;
        context.reject(
            &c.ne(mask, &context.tensor(&flags)?)?,
            "derived gate ownership mask disagrees with checked static slots",
        )?;
        context.reject(
            &c.mul(
                mask,
                &c.mul(
                    &c.ne(&derived_isolated, &context.zero)?,
                    &c.ne(&derived_isolated, &context.one)?,
                )?,
            )?,
            "derived gate isolation mask must contain only checked binary flags",
        )?;
        context.reject(
            &c.mul(
                mask,
                &c.mul(&derived_isolated, &c.ne(opening, &context.zero)?)?,
            )?,
            "derived gate closure certificate contradicts represented opening",
        )?;
        let mut slots = Vec::with_capacity(n);
        let mut supplied = packets.iter();
        let mut isolated = vec![0.0; n];
        for (index, &assigned) in flags.iter().enumerate() {
            if assigned == 1.0 {
                slots.push(CheckedOperationSlot::DerivedGate);
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
        let isolated = c.where_cond(mask, &derived_isolated, &context.tensor(&isolated)?)?;
        Ok(Self {
            slots,
            isolated,
            derived_mask: Some(mask),
            opening: Some(opening),
        })
    }
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }
    pub(crate) fn supplied(&self, index: usize) -> Option<StructureOperatingPoint> {
        match &self.slots[index] {
            CheckedOperationSlot::Supplied(packet) => Some(packet.operating_point),
            CheckedOperationSlot::DerivedGate => None,
        }
    }
    pub(crate) fn derived_opening(&self) -> Option<(&CpuTensor, &CpuTensor)> {
        self.opening.zip(self.derived_mask)
    }
    pub(crate) fn isolated(&self) -> &CpuTensor {
        &self.isolated
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
