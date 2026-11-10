//! Explicit accepted-state and bounded-step projection.

use super::projection_values as values;
use super::*;
use crate::cpu::structures::actuator::{
    CpuActuatorContext, CpuActuatorState, CpuActuatorStep, record::TensorRecord,
};
use crate::structures::StructureError;

pub(crate) fn states(
    context: &CpuActuatorContext,
    state: &CpuActuatorState,
) -> Result<Vec<GateActuatorState>, StructureError> {
    let projected = state.project()?;
    (0..context.definitions.len())
        .map(|i| values::state(context, &projected, i))
        .collect()
}
pub(crate) fn outputs(
    context: &CpuActuatorContext,
    state: &CpuActuatorState,
    time_s: f64,
) -> Result<Vec<GateActuatorOutput>, StructureError> {
    let position = state.position.project()?;
    let target = state.target.project()?;
    let phase = state.phase.project()?;
    let arrival = state.arrival.project()?;
    let active = state.active_valid.project()?;
    let index = state.active_index.project()?;
    let generation = state.generation.project()?;
    let completed = state.last_completion.valid.project()?;
    let completion_generation = state.last_completion.generation.project()?;
    let completion_episode = state.last_completion.episode.project()?;
    let effective_upper = state.last_completion.effective_upper.project()?;
    (0..context.definitions.len())
        .map(|i| {
            let d = context
                .definitions
                .get(i)
                .ok_or_else(|| StructureError::numerical("actuator output lacks its definition"))?;
            let element = context.slots.get(i).copied().ok_or_else(|| {
                StructureError::numerical("actuator output lacks its actual structure slot")
            })?;
            let position = values::position(&position, i)?;
            let phase = values::phase(values::value(&phase, i)?)?;
            let support = if position.upper_opening_m == 0.0 {
                GateActuatorSupport::ExactClosed
            } else if position.lower_opening_m > 0.0 {
                GateActuatorSupport::CertainlyPositive
            } else {
                GateActuatorSupport::ClosureUnresolved
            };
            let (arrival_lower_s, mut arrival_upper_s) = if values::value(&arrival.kind, i)? == 1.0
            {
                (
                    Some(values::value(&arrival.lower, i)?),
                    Some(values::value(&arrival.upper, i)?),
                )
            } else {
                (None, None)
            };
            if phase == GateActuatorPhase::Complete
                && values::flag(&active, i)?
                && values::flag(&completed, i)?
                && values::value(&completion_generation, i)? == values::value(&generation, i)?
                && values::value(&completion_episode, i)? == values::value(&index, i)?
            {
                arrival_upper_s = Some(values::value(&effective_upper, i)?);
            }
            Ok(GateActuatorOutput {
                element,
                actuator_id: d.actuator_id.clone(),
                structure_id: d.structure_id.clone(),
                time_s,
                target_opening_m: values::value(&target, i)?,
                position,
                phase,
                support,
                arrival_lower_s,
                arrival_upper_s,
            })
        })
        .collect()
}

pub(crate) fn step(
    context: &CpuActuatorContext,
    step: &CpuActuatorStep,
) -> Result<GateActuatorStepDiagnostics, StructureError> {
    let before = states(context, &step.before)?;
    let after = states(context, &step.after)?;
    let left = step.left.project()?;
    let forward = step.forward.project()?;
    let update = step.update.project()?;
    let completed = step.completed.project()?;
    let changed = step.changed.project()?;
    let receipts = (0..context.definitions.len())
        .map(|i| {
            let b = before.get(i).cloned().ok_or_else(|| {
                StructureError::numerical("actuator immediate step lacks before state")
            })?;
            let a = after.get(i).cloned().ok_or_else(|| {
                StructureError::numerical("actuator immediate step lacks after state")
            })?;
            let element = context.slots.get(i).copied().ok_or_else(|| {
                StructureError::numerical("actuator immediate step lacks actual structure slot")
            })?;
            Ok(GateActuatorStepReceipt {
                element,
                actuator_id: b.actuator_id.clone(),
                before: b,
                left_position: values::position(&left, i)?,
                left_forward: values::forward(&forward, i)?,
                update: values::update(context, &update, i)?,
                completed_episode: values::flag(&completed, i)?,
                target_changed: values::flag(&changed, i)?,
                after: a,
            })
        })
        .collect::<Result<Vec<_>, StructureError>>()?;
    Ok(GateActuatorStepDiagnostics {
        start_s: step.start,
        end_s: step.end,
        event: step.event,
        receipts,
    })
}
impl GateActuatorBatch {
    /// Explicitly project current actual opening and certified support.
    pub fn outputs(&self) -> Result<Vec<GateActuatorOutput>, StructureError> {
        outputs(&self.context, &self.state, self.time)
    }
    /// Explicitly project every continuation-relevant actuator value.
    pub fn states(&self) -> Result<Vec<GateActuatorState>, StructureError> {
        states(&self.context, &self.state)
    }
    /// Explicitly project the bounded last accepted interval.
    pub fn last_step(&self) -> Result<Option<GateActuatorStepDiagnostics>, StructureError> {
        self.last
            .as_ref()
            .map(|s| step(&self.context, s))
            .transpose()
    }
}
