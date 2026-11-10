//! Exhaustive unique actual-element writer admission and closed support intersection.

use super::{OperatingSourceBinding, OperatingSourceSupport};
use crate::structures::{StructureBatch, StructureError, StructureInputErrorKind};
use crate::tensor::structures::{
    TensorStructureContext,
    operating_source::{
        TensorOperatingSourceContext, TensorOperatingSourceGroup, TensorOperatingSourceKind,
        TensorPrescribedSource,
    },
};
use numr::dtype::DType;
use numr::runtime::{Device, Runtime};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

fn error(kind: StructureInputErrorKind, id: &str, message: &str) -> StructureError {
    StructureError::input(kind, format!("source={id:?}: {message}"))
}
fn compatible<R: Runtime<DType = DType>>(
    a: &Arc<TensorStructureContext<R>>,
    b: &Arc<TensorStructureContext<R>>,
    id: &str,
) -> Result<(), StructureError> {
    if a.device.is_same(&b.device)
        && (Arc::ptr_eq(a, b)
            || crate::structures::identity::same_elements(&a.elements, &b.elements))
    {
        Ok(())
    } else {
        Err(error(
            StructureInputErrorKind::Definition,
            id,
            "complete original structure context differs",
        ))
    }
}
fn intersection(
    current: &mut Option<OperatingSourceSupport>,
    next: OperatingSourceSupport,
) -> Result<(), StructureError> {
    let merged = current.map_or(next, |old| OperatingSourceSupport {
        start_s: if next.start_s > old.start_s {
            next.start_s
        } else {
            old.start_s
        },
        end_s: if next.end_s < old.end_s {
            next.end_s
        } else {
            old.end_s
        },
    });
    if merged.start_s > merged.end_s {
        return Err(StructureError::input(
            StructureInputErrorKind::Definition,
            "operating sources have no common closed support point",
        ));
    }
    *current = Some(merged);
    Ok(())
}
pub(super) fn build<R: Runtime<DType = DType>>(
    context: Arc<TensorStructureContext<R>>,
    sources: Vec<OperatingSourceBinding<'_, R>>,
) -> Result<TensorOperatingSourceContext<R>, StructureError>
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
    let elements = context
        .elements
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id.as_str(), i))
        .collect::<BTreeMap<_, _>>();
    let mut ids = BTreeSet::new();
    let mut groups = Vec::with_capacity(context.elements.len().min(sources.len()));
    let mut owner = vec![None; context.elements.len()];
    let mut support = None;
    let mut prescribed_assignments = Vec::new();
    for source in sources {
        let (id, slots, kind) = match source {
            OperatingSourceBinding::Manual {
                source_id,
                structure_ids,
            } => {
                let mut slots = Vec::with_capacity(structure_ids.len());
                for name in structure_ids {
                    let slot = elements.get(name.as_str()).copied().ok_or_else(|| {
                        error(
                            StructureInputErrorKind::Identity,
                            &source_id,
                            "manual source references an unknown original element",
                        )
                    })?;
                    if slots.last().is_some_and(|previous| *previous >= slot) {
                        return Err(error(StructureInputErrorKind::Identity,&source_id,"manual elements must be unique and follow original complete structure order").at(slot));
                    }
                    slots.push(slot);
                }
                (source_id, slots, TensorOperatingSourceKind::Manual)
            }
            OperatingSourceBinding::Prescribed {
                source_id,
                trajectory,
            } => {
                compatible(&context, &trajectory.inner.context, &source_id)?;
                let domain = trajectory.support();
                intersection(
                    &mut support,
                    OperatingSourceSupport {
                        start_s: domain.start_s,
                        end_s: domain.end_s,
                    },
                )?;
                prescribed_assignments.extend(
                    trajectory
                        .inner
                        .slots
                        .iter()
                        .copied()
                        .zip(trajectory.assignments().iter().cloned()),
                );
                (
                    source_id,
                    trajectory.inner.slots.clone(),
                    TensorOperatingSourceKind::Prescribed(TensorPrescribedSource {
                        support: OperatingSourceSupport {
                            start_s: domain.start_s,
                            end_s: domain.end_s,
                        },
                        event_times: trajectory
                            .inner
                            .events
                            .iter()
                            .map(|event| event.time_s)
                            .collect(),
                        lanes: Vec::new(),
                    }),
                )
            }
            OperatingSourceBinding::Actuated {
                source_id,
                actuator,
            } => {
                compatible(&context, &actuator.context.structures, &source_id)?;
                let domain = actuator.schedule();
                intersection(
                    &mut support,
                    OperatingSourceSupport {
                        start_s: domain.initial_time_s,
                        end_s: domain.end_s,
                    },
                )?;
                (
                    source_id,
                    actuator.context.slots.clone(),
                    TensorOperatingSourceKind::Actuated(Arc::clone(&actuator.context)),
                )
            }
            OperatingSourceBinding::ControlledPump { source_id, pump } => {
                compatible(&context, &pump.context.controls.structures, &source_id)?;
                let domain = pump.controls().schedule();
                intersection(
                    &mut support,
                    OperatingSourceSupport {
                        start_s: domain.initial_time_s,
                        end_s: domain.end_s,
                    },
                )?;
                (
                    source_id,
                    pump.context.controls.slots.clone(),
                    TensorOperatingSourceKind::ControlledPump(Arc::clone(&pump.context)),
                )
            }
        };
        if id.trim().is_empty()
            || id.chars().any(char::is_control)
            || !ids.insert(id.clone())
            || slots.is_empty()
        {
            return Err(error(
                StructureInputErrorKind::Identity,
                &id,
                "source IDs must be unique and nonempty, with nonempty ordered element groups",
            ));
        }
        let group = groups.len();
        for &slot in &slots {
            let entry = owner.get_mut(slot).ok_or_else(|| {
                error(
                    StructureInputErrorKind::Identity,
                    &id,
                    "source slot exceeds complete original context",
                )
            })?;
            if entry.replace(group).is_some() {
                return Err(error(
                    StructureInputErrorKind::Identity,
                    &id,
                    "element already has a registered source writer",
                )
                .at(slot));
            }
        }
        groups.push(TensorOperatingSourceGroup { id, slots, kind });
    }
    let owner = owner
        .into_iter()
        .enumerate()
        .map(|(slot, owner)| {
            owner.ok_or_else(|| {
                StructureError::input(
                    StructureInputErrorKind::Identity,
                    "every original structure requires one explicit source writer",
                )
                .at(slot)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    drop(elements);
    prescribed_assignments.sort_by_key(|(slot, _)| *slot);
    let prescribed = if prescribed_assignments.is_empty() {
        None
    } else {
        Some(
            StructureBatch {
                context: Arc::clone(&context),
            }
            .with_gate_trajectory_subset(
                prescribed_assignments
                    .into_iter()
                    .map(|(_, assignment)| assignment)
                    .collect(),
            )?,
        )
    };
    TensorOperatingSourceContext::new(context, groups, owner, support, prescribed)
}
