//! Immutable point-only hydraulic/source pairing without integrated-transfer APIs.

use super::{OperatingSourceDetail, OperatingSourceOutput, OperatingSourceReceipt};
use crate::cpu::structures::{
    actuator::record::TensorRecord,
    operating_source::{
        CpuOperatingSourceContext, CpuOperatingSourceKind, CpuOperatingSourcePoint,
        CpuOperatingSourceValue,
    },
};
use crate::structures::{
    GateActuatorSupport, StructureError, StructureOperation, StructureProposal,
};
use std::sync::Arc;

#[derive(Debug)]
struct PointData {
    context: Arc<CpuOperatingSourceContext>,
    time_s: f64,
    hydraulic: StructureProposal,
    point: CpuOperatingSourcePoint,
}
/// Immutable center-based point proposal paired with retained operating-source and opening evidence.
///
/// No held-volume API or implicit integrated-transfer conversion exists on this type.
#[derive(Clone, Debug)]
pub struct OperatingSourceProposal {
    inner: Arc<PointData>,
}
impl OperatingSourceProposal {
    pub(crate) fn new(
        context: Arc<CpuOperatingSourceContext>,
        time_s: f64,
        hydraulic: StructureProposal,
        point: CpuOperatingSourcePoint,
    ) -> Self {
        Self {
            inner: Arc::new(PointData {
                context,
                time_s,
                hydraulic,
                point,
            }),
        }
    }
    /// Original point query time, preserving its exact F64 bits.
    pub fn time_s(&self) -> f64 {
        self.inner.time_s
    }
    /// Explicitly project complete original operations without reevaluating any source.
    pub fn operations(&self) -> Result<Vec<StructureOperation>, StructureError> {
        self.inner.hydraulic.operations()
    }
    /// Explicitly project hydraulic center results with mandatory source support and uncertainty receipts.
    pub fn outputs(&self) -> Result<Vec<OperatingSourceOutput>, StructureError> {
        let data = &self.inner;
        let n = data.context.structures.elements.len();
        let errors = crate::structures::projection::read(&data.point.absolute_error, n)?;
        let mut details = vec![None; n];
        for (group, value) in data.context.groups.iter().zip(&data.point.values) {
            match (&group.kind, value) {
                (CpuOperatingSourceKind::Manual, CpuOperatingSourceValue::Manual) => {
                    for &slot in &group.slots {
                        details[slot] = Some(OperatingSourceDetail::Manual);
                    }
                }
                (
                    CpuOperatingSourceKind::Prescribed(source),
                    CpuOperatingSourceValue::Prescribed,
                ) => {
                    for (&slot, &lane) in group.slots.iter().zip(&source.lanes) {
                        let metadata =
                            data.point.prescribed_metadata.get(lane).ok_or_else(|| {
                                StructureError::numerical(
                                    "prescribed source lacks its retained combined point metadata",
                                )
                            })?;
                        details[slot] = Some(OperatingSourceDetail::PrescribedGate {
                            absolute_opening_error_m: errors[slot],
                            segment_index: metadata.segment_index,
                            exact_knot_index: metadata.exact_knot_index,
                            closed_by_definition: metadata.closed_by_definition,
                        });
                    }
                }
                (
                    CpuOperatingSourceKind::Actuated(context),
                    CpuOperatingSourceValue::Actuated(value),
                ) => {
                    let position = value.position.project()?;
                    let phase =
                        crate::structures::projection::read(&value.phase, group.slots.len())?;
                    for (lane, &slot) in group.slots.iter().enumerate() {
                        let position =
                            crate::structures::project_actuator_position(&position, lane)?;
                        let phase = crate::structures::project_actuator_phase(phase[lane])?;
                        let support = if position.upper_opening_m == 0.0 {
                            GateActuatorSupport::ExactClosed
                        } else if position.lower_opening_m > 0.0 {
                            GateActuatorSupport::CertainlyPositive
                        } else {
                            GateActuatorSupport::ClosureUnresolved
                        };
                        details[slot] = Some(OperatingSourceDetail::ActuatedGate {
                            actuator_id: context.definitions[lane].actuator_id.clone(),
                            accepted_steps: value.accepted_steps,
                            accepted_time_s: value.accepted_time_s,
                            position,
                            phase,
                            support,
                        });
                    }
                }
                _ => {
                    return Err(StructureError::numerical(
                        "retained operating-source point differs from its registered source kind",
                    ));
                }
            }
        }
        let hydraulic = data.hydraulic.outputs()?;
        if hydraulic.len() != n {
            return Err(StructureError::numerical(
                "operating-source hydraulic projection has incompatible row count",
            ));
        }
        hydraulic
            .into_iter()
            .enumerate()
            .map(|(element, hydraulic)| {
                let owner = data.context.owner.get(element).copied().ok_or_else(|| {
                    StructureError::numerical("operating-source output lacks its original owner")
                })?;
                let group = data.context.groups.get(owner).ok_or_else(|| {
                    StructureError::numerical(
                        "operating-source output owner lacks its registered group",
                    )
                })?;
                let detail = details
                    .get_mut(element)
                    .and_then(Option::take)
                    .ok_or_else(|| {
                        StructureError::numerical(
                            "operating-source output lacks its mandatory source receipt",
                        )
                    })?;
                Ok(OperatingSourceOutput {
                    hydraulic,
                    source: OperatingSourceReceipt {
                        source_id: group.id.clone(),
                        element,
                        structure_id: data.context.structures.elements[element].id.clone(),
                        time_s: data.time_s,
                        detail,
                    },
                })
            })
            .collect()
    }
}
