//! Original selected-literal and bounded transition/change correspondence.

use super::super::ControlledGateBatch;
use super::check::{bits, lane};
use crate::structures::{
    ControlCommand, ControlLatch, ControlState, GateActuatorState, GateTargetUpdate,
    StructureError, StructureInputErrorKind,
};

pub(super) fn selected(
    batch: &ControlledGateBatch,
    index: usize,
    latch: ControlLatch,
) -> Result<f64, StructureError> {
    let definition = batch.controls.definitions().get(index).ok_or_else(|| {
        StructureError::input(
            StructureInputErrorKind::Definition,
            "controlled gate snapshot controller lane lacks its original definition",
        )
    })?;
    let command = match latch {
        ControlLatch::Low => definition.low_command,
        ControlLatch::High => definition.high_command,
    };
    match command {
        ControlCommand::GateTarget { opening_m } => Ok(opening_m),
        _ => Err(StructureError::input(
            StructureInputErrorKind::Definition,
            "controlled gate snapshot requires original gate-target commands",
        )),
    }
}
pub(super) fn packet(
    batch: &ControlledGateBatch,
    index: usize,
    update: &GateTargetUpdate,
    target: f64,
    time: f64,
) -> Result<(), StructureError> {
    let actuator = batch.actuators.definitions().get(index).ok_or_else(|| {
        StructureError::input(
            StructureInputErrorKind::Definition,
            "controlled gate snapshot actuator lane lacks its original definition",
        )
    })?;
    lane(
        batch,
        index,
        update.actuator_id == actuator.actuator_id
            && update.structure_id == actuator.structure_id
            && bits(update.target_opening_m, target)
            && bits(update.issued_at_s, time)
            && bits(update.available_at_s, time),
        "controlled gate snapshot target packet differs from the selected original literal or endpoint timestamps",
    )
}
pub(super) fn pair(
    batch: &ControlledGateBatch,
    index: usize,
    control: &ControlState,
    actuator: &GateActuatorState,
    processed: usize,
) -> Result<(), StructureError> {
    let controller = batch.controls.definitions().get(index).ok_or_else(|| {
        StructureError::input(
            StructureInputErrorKind::Definition,
            "controlled gate snapshot controller lane lacks its original definition",
        )
    })?;
    let response = batch.actuators.definitions().get(index).ok_or_else(|| {
        StructureError::input(
            StructureInputErrorKind::Definition,
            "controlled gate snapshot actuator lane lacks its original definition",
        )
    })?;
    let initial = batch.actuators.initial_states().get(index).ok_or_else(|| {
        StructureError::input(
            StructureInputErrorKind::Definition,
            "controlled gate snapshot actuator lane lacks its original initial target",
        )
    })?;
    lane(
        batch,
        index,
        controller.structure_id == response.structure_id
            && control.controller_id == controller.controller_id
            && actuator.actuator_id == response.actuator_id,
        "controlled gate snapshot child states differ from their actual ordered gate identities",
    )?;
    let current = selected(batch, index, control.latch)?;
    let low = selected(batch, index, ControlLatch::Low)?;
    let high = selected(batch, index, ControlLatch::High)?;
    lane(
        batch,
        index,
        actuator.target_opening_m == current,
        "controlled gate snapshot actual target differs numerically from the selected sampled command",
    )?;
    let expected_generation = if low == high {
        0
    } else {
        control.transition_count
    };
    lane(
        batch,
        index,
        actuator.target_change_count == expected_generation,
        "controlled gate snapshot actuator target generation differs from numerical command transitions",
    )?;
    if expected_generation == 0 {
        lane(
            batch,
            index,
            bits(actuator.target_opening_m, initial.target_opening_m),
            "controlled gate snapshot unchanged target loses its original initial response bits",
        )?;
    } else {
        lane(
            batch,
            index,
            bits(actuator.target_opening_m, current),
            "controlled gate snapshot changed target loses the selected original command bits",
        )?;
        let transition = control.last_transition.as_ref().ok_or_else(|| {
            StructureError::input(
                StructureInputErrorKind::Definition,
                "controlled gate snapshot changed target lacks its sampled transition",
            )
        })?;
        let change = actuator.last_target_change.as_ref().ok_or_else(|| {
            StructureError::input(
                StructureInputErrorKind::Definition,
                "controlled gate snapshot changed target lacks its actual target-change receipt",
            )
        })?;
        let from = if expected_generation == 1 {
            initial.target_opening_m
        } else {
            selected(batch, index, transition.from)?
        };
        let to = selected(batch, index, transition.to)?;
        lane(
            batch,
            index,
            transition.to == control.latch
                && transition.from != transition.to
                && change.target_index == transition.decision_index
                && bits(change.time_s, transition.time_s)
                && change.target_generation == control.transition_count
                && bits(change.previous_target_opening_m, from)
                && bits(change.update.target_opening_m, to),
            "controlled gate snapshot latest target change differs from the retained sampled transition or original command bits",
        )?;
        packet(batch, index, &change.update, to, transition.time_s)?;
    }
    if processed == 0 {
        lane(
            batch,
            index,
            actuator.latest_update.is_none(),
            "controlled gate snapshot has a raw target packet before its first decision",
        )?;
    } else {
        let time=processed.checked_sub(1).and_then(|i|batch.controls.schedule().decision_times_s.get(i)).copied().ok_or_else(||StructureError::input(StructureInputErrorKind::Definition,"controlled gate snapshot processed decision index exceeds its original calendar"))?;
        let update = actuator.latest_update.as_ref().ok_or_else(|| {
            StructureError::input(
                StructureInputErrorKind::Definition,
                "controlled gate snapshot processed decisions lack their latest raw target packet",
            )
        })?;
        packet(batch, index, update, current, time)?;
    }
    lane(
        batch,
        index,
        processed >= control.transition_count,
        "controlled gate snapshot transitions exceed processed decisions",
    )
}
