//! Cross-child clocks, calendar progress, and current continuation admission.

use super::super::ControlledGateBatch;
use super::ControlledGateSnapshotData;
use crate::structures::{
    GateActuatorBatch, OperatingControlBatch, StructureError, StructureInputErrorKind,
};

pub(super) fn bits(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}
pub(super) fn require(condition: bool, message: &str) -> Result<(), StructureError> {
    if condition {
        Ok(())
    } else {
        Err(StructureError::input(
            StructureInputErrorKind::Definition,
            message,
        ))
    }
}
pub(super) fn lane(
    batch: &ControlledGateBatch,
    index: usize,
    condition: bool,
    message: &str,
) -> Result<(), StructureError> {
    require(condition, message).map_err(|error| match batch.actuators.context.slots.get(index) {
        Some(&slot) => error.at(slot),
        None => error,
    })
}
pub(super) fn metadata(
    controls: &OperatingControlBatch,
    actuators: &GateActuatorBatch,
) -> Result<(), StructureError> {
    require(
        bits(controls.time(), actuators.time())
            && controls.accepted_steps() == actuators.accepted_steps()
            && controls.next_decision == actuators.next_target,
        "controlled gate snapshot child clocks, interval counts, or processed calendars differ",
    )
}
pub(super) fn data(
    batch: &ControlledGateBatch,
    data: &ControlledGateSnapshotData,
) -> Result<(), StructureError> {
    let controls = &data.controls;
    let actuators = &data.actuators;
    require(
        bits(controls.time_s, actuators.time_s)
            && controls.accepted_steps == actuators.accepted_steps
            && controls.next_decision_index == actuators.next_target_index,
        "controlled gate snapshot child clocks, interval counts, or processed calendars differ",
    )?;
    require(
        controls.states.len() == batch.controls.definitions().len()
            && actuators.states.len() == batch.actuators.definitions().len()
            && controls.states.len() == actuators.states.len(),
        "controlled gate snapshot requires complete matching ordered child states",
    )?;
    for (index, (control, actuator)) in controls.states.iter().zip(&actuators.states).enumerate() {
        super::state::pair(
            batch,
            index,
            control,
            actuator,
            controls.next_decision_index,
        )?;
    }
    super::interval::check(batch, data)
}
