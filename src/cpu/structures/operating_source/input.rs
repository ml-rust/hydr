//! Bit-preserving tensor assembly from prescribed positions and trusted actual responses.

use super::frames::metadata;
use super::{CpuOperatingSourceContext, CpuOperatingSourceKind};
use crate::cpu::{
    CpuTensor,
    structures::{actuator::CpuActuatorPosition, gate::trajectory::TrajectoryPointMetadata},
};
use crate::structures::{OperatingSourceFrame, StructureError};
use numr::ops::{BinaryOps, CompareOps, IndexingOps, ShapeOps};

#[derive(Clone, Debug)]
pub(crate) enum CpuOperatingSourceValue {
    Manual,
    Prescribed,
    Actuated(Box<CpuActuatedSourceValue>),
    ControlledPump(Box<CpuControlledPumpSourceValue>),
}
#[derive(Clone, Debug)]
pub(crate) struct CpuActuatedSourceValue {
    pub accepted_steps: usize,
    pub accepted_time_s: f64,
    pub position: CpuActuatorPosition,
    pub phase: CpuTensor,
}
#[derive(Clone, Debug)]
pub(crate) struct CpuControlledPumpSourceValue {
    pub accepted_steps: usize,
    pub accepted_time_s: f64,
    pub running: CpuTensor,
    pub speed: CpuTensor,
    pub high: CpuTensor,
}
#[derive(Clone, Debug)]
pub(crate) struct CpuOperatingSourcePoint {
    pub opening: CpuTensor,
    pub absolute_error: CpuTensor,
    pub closed: CpuTensor,
    pub pump_running: CpuTensor,
    pub pump_isolated: CpuTensor,
    pub pump_speed: CpuTensor,
    pub pump_power_of_two: CpuTensor,
    pub values: Vec<CpuOperatingSourceValue>,
    pub prescribed_metadata: Vec<TrajectoryPointMetadata>,
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
        pump_running: structures.zero.clone(),
        pump_isolated: structures.zero.clone(),
        pump_speed: structures.one.clone(),
        pump_power_of_two: structures.one.clone(),
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
            "dynamic source concatenation requires a nonempty checked tensor group",
        )),
        [one] => Ok((*one).clone()),
        _ => Ok(context.structures.client.cat(values, 0)?),
    }
}
pub(crate) fn assemble(
    context: &CpuOperatingSourceContext,
    time_s: f64,
    supplied_frames: &[OperatingSourceFrame<'_>],
) -> Result<CpuOperatingSourcePoint, StructureError> {
    let frames = super::frames(context, time_s, supplied_frames)?;
    let mut point = prescribed(context, time_s)?;
    let mut centers = Vec::with_capacity(frames.len());
    let mut errors = Vec::with_capacity(frames.len());
    let mut lower = Vec::with_capacity(frames.len());
    let mut upper = Vec::with_capacity(frames.len());
    let mut next = frames.into_iter();
    let mut pumps = Vec::new();
    for group in &context.groups {
        match &group.kind {
            CpuOperatingSourceKind::Manual => point.values.push(CpuOperatingSourceValue::Manual),
            CpuOperatingSourceKind::Prescribed(_) => {
                point.values.push(CpuOperatingSourceValue::Prescribed)
            }
            CpuOperatingSourceKind::Actuated(_) => {
                let Some(OperatingSourceFrame::ActuatedGate { actuator, .. }) = next.next() else {
                    return Err(StructureError::numerical(
                        "checked actuator frame vector is incomplete or has another kind",
                    ));
                };
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
            CpuOperatingSourceKind::ControlledPump(_) => {
                let Some(OperatingSourceFrame::ControlledPump { pump, .. }) = next.next() else {
                    return Err(StructureError::numerical(
                        "checked pump frame vector is incomplete or has another kind",
                    ));
                };
                let selected = pump.selected()?;
                for tensor in [
                    &selected.running,
                    &selected.speed,
                    &selected.power_of_two,
                    &selected.high,
                ] {
                    metadata(context, tensor, group.slots.len(), &group.id)?;
                }
                point
                    .values
                    .push(CpuOperatingSourceValue::ControlledPump(Box::new(
                        CpuControlledPumpSourceValue {
                            accepted_steps: pump.accepted_steps(),
                            accepted_time_s: pump.time(),
                            running: selected.running.clone(),
                            speed: selected.speed.clone(),
                            high: selected.high.clone(),
                        },
                    )));
                pumps.push(selected);
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
    if !pumps.is_empty() {
        let running = concatenate(
            context,
            &pumps.iter().map(|p| &p.running).collect::<Vec<_>>(),
        )?;
        let speed = concatenate(context, &pumps.iter().map(|p| &p.speed).collect::<Vec<_>>())?;
        let power = concatenate(
            context,
            &pumps.iter().map(|p| &p.power_of_two).collect::<Vec<_>>(),
        )?;
        let c = &context.structures.client;
        let isolated = c.sub(&context.pump_one, &running)?;
        point.pump_running =
            c.index_put(&point.pump_running, 0, &context.pump_indices, &running)?;
        point.pump_isolated =
            c.index_put(&point.pump_isolated, 0, &context.pump_indices, &isolated)?;
        point.pump_speed = c.index_put(&point.pump_speed, 0, &context.pump_indices, &speed)?;
        point.pump_power_of_two =
            c.index_put(&point.pump_power_of_two, 0, &context.pump_indices, &power)?;
    }
    Ok(point)
}
