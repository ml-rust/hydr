//! Ordered dynamic source identity, continuation, and tensor-layout admission.

use super::{CpuOperatingSourceContext, CpuOperatingSourceKind};
use crate::cpu::CpuTensor;
use crate::structures::{OperatingSourceFrame, StructureError, StructureInputErrorKind};
use numr::{dtype::DType, runtime::Device};

pub(crate) fn frames<'a>(
    context: &CpuOperatingSourceContext,
    time_s: f64,
    frames: &[OperatingSourceFrame<'a>],
) -> Result<Vec<OperatingSourceFrame<'a>>, StructureError> {
    let mut result = Vec::with_capacity(context.groups.len().min(frames.len()));
    let mut supplied = frames.iter();
    for group in &context.groups {
        if matches!(
            group.kind,
            CpuOperatingSourceKind::Manual | CpuOperatingSourceKind::Prescribed(_)
        ) {
            continue;
        }
        let frame = supplied.next().ok_or_else(|| {
            StructureError::input(
                StructureInputErrorKind::Identity,
                format!(
                    "source={:?}: supply every dynamic frame in registration order",
                    group.id
                ),
            )
        })?;
        let id = match frame {
            OperatingSourceFrame::ActuatedGate { source_id, .. }
            | OperatingSourceFrame::ControlledPump { source_id, .. } => *source_id,
        };
        if id != group.id {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                format!(
                    "source={:?},frame={id:?}: match registered dynamic source order",
                    group.id
                ),
            ));
        }
        let accepted_time = match (&group.kind, frame) {
            (
                CpuOperatingSourceKind::Actuated(original),
                OperatingSourceFrame::ActuatedGate { actuator, .. },
            ) => {
                if !crate::structures::same_actuator_context(original, &actuator.context) {
                    return Err(incompatible(
                        &group.id,
                        "actuator immutable context, original inputs, or policies differ",
                    ));
                }
                actuator.time()
            }
            (
                CpuOperatingSourceKind::ControlledPump(original),
                OperatingSourceFrame::ControlledPump { pump, .. },
            ) => {
                if !crate::structures::same_controlled_pump_context(original, &pump.context) {
                    return Err(incompatible(
                        &group.id,
                        "pump response or original control context differs",
                    ));
                }
                pump.time()
            }
            _ => {
                return Err(StructureError::input(
                    StructureInputErrorKind::Identity,
                    format!(
                        "source={:?}: supply the registered dynamic response kind",
                        group.id
                    ),
                ));
            }
        };
        if accepted_time != time_s {
            return Err(StructureError::input(
                StructureInputErrorKind::UnsupportedRegime,
                format!(
                    "source={:?},accepted_time={accepted_time},query_time={time_s}: supply a response frame at the exact query time",
                    group.id
                ),
            ));
        }
        match frame {
            OperatingSourceFrame::ActuatedGate { actuator, .. } => {
                for tensor in [
                    &actuator.state.position.center,
                    &actuator.state.position.lower,
                    &actuator.state.position.upper,
                    &actuator.state.position.error,
                    &actuator.state.phase,
                ] {
                    metadata(context, tensor, group.slots.len(), &group.id)?;
                }
            }
            OperatingSourceFrame::ControlledPump { pump, .. } => {
                metadata(
                    context,
                    &pump.controls.state.high,
                    group.slots.len(),
                    &group.id,
                )?;
            }
        }
        result.push(*frame);
    }
    if supplied.next().is_some() {
        return Err(StructureError::input(
            StructureInputErrorKind::Identity,
            "extra operating source frame: supply only registered dynamic groups",
        ));
    }
    Ok(result)
}

fn incompatible(id: &str, message: &str) -> StructureError {
    StructureError::input(
        StructureInputErrorKind::Definition,
        format!("source={id:?}: {message}"),
    )
}

pub(super) fn metadata(
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
