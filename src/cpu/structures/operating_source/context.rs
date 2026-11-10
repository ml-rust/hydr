//! Cached original source slots, replacement indices, and immutable provider contexts.

use crate::cpu::{
    CpuTensor,
    structures::{CpuStructureContext, actuator::CpuActuatorContext},
};
use crate::structures::{GateTrajectoryBatch, OperatingSourceSupport, StructureError};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) enum CpuOperatingSourceKind {
    Manual,
    Prescribed(CpuPrescribedSource),
    Actuated(Arc<CpuActuatorContext>),
}
#[derive(Debug)]
pub(crate) struct CpuPrescribedSource {
    pub support: OperatingSourceSupport,
    pub event_times: Vec<f64>,
    pub lanes: Vec<usize>,
}
#[derive(Debug)]
pub(crate) struct CpuOperatingSourceGroup {
    pub id: String,
    pub slots: Vec<usize>,
    pub kind: CpuOperatingSourceKind,
}
#[derive(Debug)]
pub(crate) struct CpuOperatingSourceContext {
    pub structures: Arc<CpuStructureContext>,
    pub groups: Vec<CpuOperatingSourceGroup>,
    pub owner: Vec<usize>,
    pub manual_slots: Vec<usize>,
    pub derived_slots: Vec<usize>,
    pub derived_mask: CpuTensor,
    pub support: Option<OperatingSourceSupport>,
    pub prescribed: Option<GateTrajectoryBatch>,
    pub actuator_indices: CpuTensor,
    pub actuator_zero: CpuTensor,
    pub actuator_one: CpuTensor,
}
impl CpuOperatingSourceContext {
    pub(crate) fn new(
        structures: Arc<CpuStructureContext>,
        mut groups: Vec<CpuOperatingSourceGroup>,
        owner: Vec<usize>,
        support: Option<OperatingSourceSupport>,
        prescribed: Option<GateTrajectoryBatch>,
    ) -> Result<Self, StructureError> {
        let mut mask = vec![0.0; structures.elements.len()];
        let mut manual_slots = Vec::new();
        let mut derived_slots = Vec::new();
        for (slot, &group) in owner.iter().enumerate() {
            match groups.get(group).map(|g| &g.kind) {
                Some(CpuOperatingSourceKind::Manual) => manual_slots.push(slot),
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
        let mut prescribed_lanes = vec![None; structures.elements.len()];
        if let Some(binding) = &prescribed {
            for (lane, &slot) in binding.inner.slots.iter().enumerate() {
                prescribed_lanes[slot] = Some(lane);
            }
        }
        let mut actuator_slots = Vec::new();
        for group in &mut groups {
            match &mut group.kind {
                CpuOperatingSourceKind::Prescribed(source) => {
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
                CpuOperatingSourceKind::Actuated(_) => {
                    actuator_slots.extend_from_slice(&group.slots)
                }
                CpuOperatingSourceKind::Manual => {}
            }
        }
        let actuator_indices =
            super::super::gate::trajectory::indices(&structures, &actuator_slots)?;
        let actuator_zero = structures.tensor(&vec![0.0; actuator_slots.len()])?;
        let actuator_one = structures.tensor(&vec![1.0; actuator_slots.len()])?;
        Ok(Self {
            structures,
            groups,
            owner,
            manual_slots,
            derived_slots,
            derived_mask,
            support,
            prescribed,
            actuator_indices,
            actuator_zero,
            actuator_one,
        })
    }
}
