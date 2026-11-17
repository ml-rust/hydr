//! Complete original gate authority and common-calendar admission.

use crate::structures::identity::{bits, same_elements};
use crate::structures::{
    ControlCommand, ControlLatch, GateActuatorBatch, OperatingControlBatch, StructureError,
    StructureInputErrorKind,
};
use numr::dtype::DType;
use numr::runtime::{Device, Runtime};
use std::sync::Arc;

pub(super) struct Binding {
    pub low_targets: Vec<f64>,
    pub high_targets: Vec<f64>,
}

pub(super) fn check<R: Runtime<DType = DType>>(
    controls: &OperatingControlBatch<R>,
    actuators: &GateActuatorBatch<R>,
) -> Result<Binding, StructureError> {
    let invalid = |message| StructureError::input(StructureInputErrorKind::Definition, message);
    if controls.steps != 0
        || actuators.steps != 0
        || controls.next_decision != 0
        || actuators.next_target != 0
        || controls.last.is_some()
        || actuators.last.is_some()
    {
        return Err(invalid(
            "controlled gate binding requires fresh initial children: restore a composite snapshot to resume",
        ));
    }
    if !controls
        .context
        .structures
        .device
        .is_same(&actuators.context.structures.device)
    {
        return Err(invalid(
            "controlled gate children use different device contexts: bind both on the same configured device",
        ));
    }
    if !Arc::ptr_eq(&controls.context.structures, &actuators.context.structures)
        && !same_elements(controls.elements(), actuators.elements())
    {
        return Err(StructureError::input(
            StructureInputErrorKind::Identity,
            "controlled gate models differ: supply children from the same complete actual structure model",
        ));
    }
    let decisions = controls.schedule();
    let targets = actuators.schedule();
    if !bits(
        &[decisions.initial_time_s, decisions.end_s],
        &[targets.initial_time_s, targets.end_s],
    ) || !bits(&decisions.decision_times_s, &targets.target_times_s)
    {
        return Err(invalid(
            "controlled gate calendars differ: supply identical original initial, decision, and terminal times",
        ));
    }
    if controls.definitions().len() != actuators.definitions().len() {
        return Err(StructureError::input(
            StructureInputErrorKind::Identity,
            "controlled gate authority is incomplete: supply one controller per supplied actuator",
        ));
    }
    let mut low_targets = Vec::with_capacity(actuators.definitions().len());
    let mut high_targets = Vec::with_capacity(actuators.definitions().len());
    for (lane, (definition, actuator)) in controls
        .definitions()
        .iter()
        .zip(actuators.definitions())
        .enumerate()
    {
        let slot = actuators.context.slots[lane];
        if definition.structure_id != actuator.structure_id || controls.context.slots[lane] != slot
        {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                "controlled gate subsets differ: bind identical ordered actual gate subsets",
            )
            .at(slot));
        }
        let (low, high) = match (definition.low_command, definition.high_command) {
            (
                ControlCommand::GateTarget { opening_m: low },
                ControlCommand::GateTarget { opening_m: high },
            ) => (low, high),
            _ => {
                return Err(invalid(
                    "controlled gate commands include another device kind: supply only original gate-target commands",
                )
                .at(slot));
            }
        };
        let selected = match controls.initial_states()[lane].latch {
            ControlLatch::Low => low,
            ControlLatch::High => high,
        };
        if selected != actuators.initial_states()[lane].target_opening_m {
            return Err(invalid(
                "initial control command differs from actuator target: supply numerically equal original targets",
            )
            .at(slot));
        }
        low_targets.push(low);
        high_targets.push(high);
    }
    Ok(Binding {
        low_targets,
        high_targets,
    })
}
