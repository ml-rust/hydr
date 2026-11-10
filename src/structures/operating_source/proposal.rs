//! Immutable point-only hydraulic/source pairing without integrated-transfer APIs.

use super::{OperatingSourceDetail, OperatingSourceOutput, OperatingSourceReceipt};
use crate::structures::{
    ControlLatch, GateActuatorSupport, StructureError, StructureOperatingPoint, StructureOperation,
    StructureProposal,
};
use crate::tensor::structures::{
    actuator::record::TensorRecord,
    operating_source::{
        TensorOperatingSourceContext, TensorOperatingSourceKind, TensorOperatingSourcePoint,
        TensorOperatingSourceValue,
    },
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

#[derive(Debug)]
struct PointData<R: Runtime<DType = DType>> {
    context: Arc<TensorOperatingSourceContext<R>>,
    time_s: f64,
    hydraulic: StructureProposal<R>,
    point: TensorOperatingSourcePoint<R>,
}
/// Immutable point proposal paired with retained operation provenance and gate-opening evidence.
///
/// No held-volume API or implicit integrated-transfer conversion exists on this type.
#[derive(Clone, Debug)]
pub struct OperatingSourceProposal<R: Runtime<DType = DType>> {
    inner: Arc<PointData<R>>,
}
impl<R: Runtime<DType = DType>> OperatingSourceProposal<R> {
    pub(crate) fn new(
        context: Arc<TensorOperatingSourceContext<R>>,
        time_s: f64,
        hydraulic: StructureProposal<R>,
        point: TensorOperatingSourcePoint<R>,
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
                (TensorOperatingSourceKind::Manual, TensorOperatingSourceValue::Manual) => {
                    for &slot in &group.slots {
                        details[slot] = Some(OperatingSourceDetail::Manual);
                    }
                }
                (
                    TensorOperatingSourceKind::Prescribed(source),
                    TensorOperatingSourceValue::Prescribed,
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
                    TensorOperatingSourceKind::Actuated(context),
                    TensorOperatingSourceValue::Actuated(value),
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
                (
                    TensorOperatingSourceKind::ControlledPump(context),
                    TensorOperatingSourceValue::ControlledPump(value),
                ) => {
                    let count = group.slots.len();
                    let running = crate::structures::projection::read(&value.running, count)?;
                    let speed = crate::structures::projection::read(&value.speed, count)?;
                    let high = crate::structures::projection::read(&value.high, count)?;
                    for (lane, &slot) in group.slots.iter().enumerate() {
                        let operating_point = if running[lane] == 0.0 {
                            StructureOperatingPoint::PumpStoppedIsolated
                        } else {
                            StructureOperatingPoint::PumpRunning {
                                speed_ratio: speed[lane],
                            }
                        };
                        details[slot] = Some(OperatingSourceDetail::ControlledPump {
                            controller_id: context.controls.definitions[lane].controller_id.clone(),
                            accepted_steps: value.accepted_steps,
                            accepted_time_s: value.accepted_time_s,
                            latch: if high[lane] == 0.0 {
                                ControlLatch::Low
                            } else {
                                ControlLatch::High
                            },
                            response_model: context.definitions[lane].model,
                            operating_point,
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
