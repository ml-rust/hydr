//! Cached original source slots, replacement indices, and immutable provider contexts.

use crate::structures::{GateTrajectoryBatch, OperatingSourceSupport, StructureError};
use crate::tensor::structures::{
    TensorStructureContext, actuator::TensorActuatorContext,
    controlled_pump::TensorControlledPumpContext,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) enum TensorOperatingSourceKind<R: Runtime<DType = DType>> {
    Manual,
    Prescribed(TensorPrescribedSource),
    Actuated(Arc<TensorActuatorContext<R>>),
    ControlledPump(Arc<TensorControlledPumpContext<R>>),
}
#[derive(Debug)]
pub(crate) struct TensorPrescribedSource {
    pub support: OperatingSourceSupport,
    pub event_times: Vec<f64>,
    pub lanes: Vec<usize>,
}
#[derive(Debug)]
pub(crate) struct TensorOperatingSourceGroup<R: Runtime<DType = DType>> {
    pub id: String,
    pub slots: Vec<usize>,
    pub kind: TensorOperatingSourceKind<R>,
}
#[derive(Debug)]
pub(crate) struct TensorOperatingSourceContext<R: Runtime<DType = DType>> {
    pub structures: Arc<TensorStructureContext<R>>,
    pub groups: Vec<TensorOperatingSourceGroup<R>>,
    pub owner: Vec<usize>,
    pub manual_slots: Vec<usize>,
    pub derived_slots: Vec<usize>,
    pub derived_mask: Tensor<R>,
    pub pump_slots: Vec<usize>,
    pub pump_mask: Tensor<R>,
    pub support: Option<OperatingSourceSupport>,
    pub prescribed: Option<GateTrajectoryBatch<R>>,
    pub actuator_indices: Tensor<R>,
    pub actuator_zero: Tensor<R>,
    pub actuator_one: Tensor<R>,
    pub pump_indices: Tensor<R>,
    pub pump_one: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorOperatingSourceContext<R> {
    pub(crate) fn new(
        structures: Arc<TensorStructureContext<R>>,
        mut groups: Vec<TensorOperatingSourceGroup<R>>,
        owner: Vec<usize>,
        support: Option<OperatingSourceSupport>,
        prescribed: Option<GateTrajectoryBatch<R>>,
    ) -> Result<Self, StructureError> {
        let mut mask = vec![0.0; structures.elements.len()];
        let mut manual_slots = Vec::new();
        let mut derived_slots = Vec::new();
        let mut pump_slots = Vec::new();
        let mut pump_mask = vec![0.0; structures.elements.len()];
        for (slot, &group) in owner.iter().enumerate() {
            match groups.get(group).map(|g| &g.kind) {
                Some(TensorOperatingSourceKind::Manual) => manual_slots.push(slot),
                Some(TensorOperatingSourceKind::ControlledPump(_)) => {
                    pump_slots.push(slot);
                    pump_mask[slot] = 1.0;
                }
                Some(_) => {
                    derived_slots.push(slot);
                    mask[slot] = 1.0;
                }
                None => {
                    return Err(StructureError::numerical(
                        "operating source owner has no registered group",
                    ));
                }
            }
        }
        let derived_mask = structures.tensor(&mask)?;
        let pump_mask = structures.tensor(&pump_mask)?;
        let mut prescribed_lanes = vec![None; structures.elements.len()];
        if let Some(binding) = &prescribed {
            for (lane, &slot) in binding.inner.slots.iter().enumerate() {
                prescribed_lanes[slot] = Some(lane);
            }
        }
        let mut actuator_slots = Vec::new();
        let mut pump_group_slots = Vec::new();
        for group in &mut groups {
            match &mut group.kind {
                TensorOperatingSourceKind::Prescribed(source) => {
                    source.lanes = group
                        .slots
                        .iter()
                        .map(|&slot| {
                            prescribed_lanes[slot].ok_or_else(|| {
                                StructureError::numerical(
                                    "prescribed source lacks its checked combined assignment",
                                )
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                }
                TensorOperatingSourceKind::Actuated(_) => {
                    actuator_slots.extend_from_slice(&group.slots)
                }
                TensorOperatingSourceKind::ControlledPump(_) => {
                    pump_group_slots.extend_from_slice(&group.slots)
                }
                TensorOperatingSourceKind::Manual => {}
            }
        }
        let actuator_indices =
            super::super::gate::trajectory::indices(&structures, &actuator_slots)?;
        let actuator_zero = structures.tensor(&vec![0.0; actuator_slots.len()])?;
        let actuator_one = structures.tensor(&vec![1.0; actuator_slots.len()])?;
        let pump_indices = super::super::gate::trajectory::indices(&structures, &pump_group_slots)?;
        let pump_one = structures.tensor(&vec![1.0; pump_group_slots.len()])?;
        Ok(Self {
            structures,
            groups,
            owner,
            manual_slots,
            derived_slots,
            derived_mask,
            pump_slots,
            pump_mask,
            support,
            prescribed,
            actuator_indices,
            actuator_zero,
            actuator_one,
            pump_indices,
            pump_one,
        })
    }
}
