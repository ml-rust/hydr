//! Bit-preserving tensor replacement from original-time or trusted actual positions.

use super::{CpuOperatingSourceContext, CpuOperatingSourceKind};
use crate::cpu::{
    CpuTensor,
    structures::{actuator::CpuActuatorPosition, gate::trajectory::TrajectoryPointMetadata},
};
use crate::structures::{
    ActuatorSourceFrame, GateActuatorBatch, StructureError, StructureInputErrorKind,
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, IndexingOps, ShapeOps},
    runtime::Device,
};

#[derive(Clone, Debug)]
pub(crate) enum CpuOperatingSourceValue {
    Manual,
    Prescribed,
    Actuated(Box<CpuActuatedSourceValue>),
}
#[derive(Clone, Debug)]
pub(crate) struct CpuActuatedSourceValue {
    pub accepted_steps: usize,
    pub accepted_time_s: f64,
    pub position: CpuActuatorPosition,
    pub phase: CpuTensor,
}
#[derive(Clone, Debug)]
pub(crate) struct CpuOperatingSourcePoint {
    pub opening: CpuTensor,
    pub absolute_error: CpuTensor,
    pub closed: CpuTensor,
    pub values: Vec<CpuOperatingSourceValue>,
    pub prescribed_metadata: Vec<TrajectoryPointMetadata>,
}
pub(crate) fn frames<'a>(
    context: &CpuOperatingSourceContext,
    time_s: f64,
    frames: &[ActuatorSourceFrame<'a>],
) -> Result<Vec<&'a GateActuatorBatch>, StructureError> {
    let mut result = Vec::with_capacity(context.groups.len().min(frames.len()));
    let mut supplied = frames.iter();
    for group in &context.groups {
        if let CpuOperatingSourceKind::Actuated(original) = &group.kind {
            let frame = supplied.next().ok_or_else(|| {
                StructureError::input(
                    StructureInputErrorKind::Identity,
                    format!(
                        "source={:?}: supply every actuated frame in registration order",
                        group.id
                    ),
                )
            })?;
            if frame.source_id != group.id {
                return Err(StructureError::input(
                    StructureInputErrorKind::Identity,
                    format!(
                        "source={:?},frame={:?}: match registered actuated source order",
                        group.id, frame.source_id
                    ),
                ));
            }
            if !crate::structures::same_actuator_context(original, &frame.actuator.context) {
                return Err(StructureError::input(
                    StructureInputErrorKind::Definition,
                    format!(
                        "source={:?}: actuator immutable context, original inputs, or policies differ",
                        group.id
                    ),
                ));
            }
            if frame.actuator.time() != time_s {
                return Err(StructureError::input(
                    StructureInputErrorKind::UnsupportedRegime,
                    format!(
                        "source={:?},accepted_time={},query_time={time_s}: supply an actuator frame at the exact query time",
                        group.id,
                        frame.actuator.time()
                    ),
                ));
            }
            for tensor in [
                &frame.actuator.state.position.center,
                &frame.actuator.state.position.lower,
                &frame.actuator.state.position.upper,
                &frame.actuator.state.position.error,
                &frame.actuator.state.phase,
            ] {
                metadata(context, tensor, group.slots.len(), &group.id)?;
            }
            result.push(frame.actuator);
        }
    }
    if supplied.next().is_some() {
        return Err(StructureError::input(
            StructureInputErrorKind::Identity,
            "extra actuator source frame: supply only registered actuated groups",
        ));
    }
    Ok(result)
}
fn metadata(
    context: &CpuOperatingSourceContext,
    tensor: &CpuTensor,
    count: usize,
    id: &str,
) -> Result<(), StructureError> {
    if tensor.dtype() != DType::F64
        || tensor.shape() != [count]
        || tensor.device().id() != context.structures.device.id()
    {
        return Err(StructureError::input(
            StructureInputErrorKind::Definition,
            format!(
                "source={id:?}: require matching CPU F64 device and ordered source tensor shape"
            ),
        ));
    }
    Ok(())
}
fn prescribed(
    context: &CpuOperatingSourceContext,
    time_s: f64,
) -> Result<CpuOperatingSourcePoint, StructureError> {
    let structures = &context.structures;
    let mut point = CpuOperatingSourcePoint {
        opening: structures.zero.clone(),
        absolute_error: structures.zero.clone(),
        closed: structures.zero.clone(),
        values: Vec::with_capacity(context.groups.len()),
        prescribed_metadata: Vec::new(),
    };
    let Some(binding) = &context.prescribed else {
        return Ok(point);
    };
    let evaluation = binding.evaluate(time_s)?;
    let data = &evaluation.inner;
    if data.metadata.len() != binding.inner.slots.len() {
        return Err(StructureError::numerical(
            "combined prescribed metadata differs from its checked assignment subset",
        ));
    }
    metadata(
        context,
        &data.opening,
        structures.elements.len(),
        "combined prescribed sources",
    )?;
    metadata(
        context,
        &data.absolute_error,
        structures.elements.len(),
        "combined prescribed sources",
    )?;
    let mut original = vec![0.0; structures.elements.len()];
    for (&slot, metadata) in binding.inner.slots.iter().zip(&data.metadata) {
        original[slot] = f64::from(metadata.closed_by_definition);
    }
    let original = structures.tensor(&original)?;
    let c = &structures.client;
    let canonical = c.mul(
        &c.eq(&data.opening, &structures.zero)?,
        &c.eq(&data.absolute_error, &structures.zero)?,
    )?;
    structures.reject(
        &c.mul(&original, &c.sub(&structures.one, &canonical)?)?,
        "prescribed source closure differs from canonical zero position/error",
    )?;
    point.opening = data.opening.clone();
    point.absolute_error = data.absolute_error.clone();
    point.closed = c.mul(&original, &canonical)?;
    point.prescribed_metadata = data.metadata.clone();
    Ok(point)
}
fn concatenate(
    context: &CpuOperatingSourceContext,
    values: &[&CpuTensor],
) -> Result<CpuTensor, StructureError> {
    match values {
        [] => Err(StructureError::numerical(
            "actuator source concatenation requires a nonempty checked tensor group",
        )),
        [one] => Ok((*one).clone()),
        _ => Ok(context.structures.client.cat(values, 0)?),
    }
}
pub(crate) fn assemble(
    context: &CpuOperatingSourceContext,
    time_s: f64,
    supplied_frames: &[ActuatorSourceFrame<'_>],
) -> Result<CpuOperatingSourcePoint, StructureError> {
    let frames = frames(context, time_s, supplied_frames)?;
    let mut point = prescribed(context, time_s)?;
    let mut centers = Vec::with_capacity(frames.len());
    let mut errors = Vec::with_capacity(frames.len());
    let mut lower = Vec::with_capacity(frames.len());
    let mut upper = Vec::with_capacity(frames.len());
    let mut next = frames.into_iter();
    for group in &context.groups {
        match &group.kind {
            CpuOperatingSourceKind::Manual => point.values.push(CpuOperatingSourceValue::Manual),
            CpuOperatingSourceKind::Prescribed(_) => {
                point.values.push(CpuOperatingSourceValue::Prescribed)
            }
            CpuOperatingSourceKind::Actuated(_) => {
                let actuator = next.next().ok_or_else(|| {
                    StructureError::numerical("checked actuator frame vector is incomplete")
                })?;
                let position = &actuator.state.position;
                centers.push(&position.center);
                errors.push(&position.error);
                lower.push(&position.lower);
                upper.push(&position.upper);
                point
                    .values
                    .push(CpuOperatingSourceValue::Actuated(Box::new(
                        CpuActuatedSourceValue {
                            accepted_steps: actuator.accepted_steps(),
                            accepted_time_s: actuator.time(),
                            position: position.clone(),
                            phase: actuator.state.phase.clone(),
                        },
                    )));
            }
        }
    }
    if !centers.is_empty() {
        let center = concatenate(context, &centers)?;
        let error = concatenate(context, &errors)?;
        let lower = concatenate(context, &lower)?;
        let upper = concatenate(context, &upper)?;
        let c = &context.structures.client;
        let zero = &context.actuator_zero;
        let canonical = c.mul(
            &c.eq(&center, zero)?,
            &c.mul(
                &c.eq(&lower, zero)?,
                &c.mul(&c.eq(&upper, zero)?, &c.eq(&error, zero)?)?,
            )?,
        )?;
        let invalid = c.mul(
            &c.eq(&center, zero)?,
            &c.sub(&context.actuator_one, &canonical)?,
        )?;
        if context.structures.any(&invalid)? {
            let invalid = c.index_put(
                &context.structures.zero,
                0,
                &context.actuator_indices,
                &invalid,
            )?;
            context.structures.reject(
                &invalid,
                "actuated source zero center lacks canonical closed support",
            )?;
        }
        point.opening = c.index_put(&point.opening, 0, &context.actuator_indices, &center)?;
        point.absolute_error =
            c.index_put(&point.absolute_error, 0, &context.actuator_indices, &error)?;
        point.closed = c.index_put(&point.closed, 0, &context.actuator_indices, &canonical)?;
    }
    Ok(point)
}
