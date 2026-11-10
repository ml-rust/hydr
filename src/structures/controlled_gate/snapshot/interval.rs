//! Immediate before-state and endpoint decision correspondence across both children.

use super::super::ControlledGateBatch;
use super::ControlledGateSnapshotData;
use super::check::{bits, lane, require};
use crate::structures::{StructureError, StructureInputErrorKind};

pub(super) fn check(
    batch: &ControlledGateBatch,
    data: &ControlledGateSnapshotData,
) -> Result<(), StructureError> {
    match (&data.controls.last_step, &data.actuators.last_step) {
        (None, None) => Ok(()),
        (Some(control), Some(actuator)) => {
            require(
                bits(control.start_s, actuator.start_s) && bits(control.end_s, actuator.end_s),
                "controlled gate snapshot children retain different immediate interval endpoints",
            )?;
            let decision = actuator.event.and_then(|event| event.target_index);
            require(
                control.decision_index == decision,
                "controlled gate snapshot immediate decision occurrence differs between children",
            )?;
            require(
                actuator.receipts.len() == data.actuators.states.len(),
                "controlled gate snapshot immediate actuator receipts are incomplete",
            )?;
            require(
                if control.decision_index.is_some() {
                    control.decisions.len() == data.controls.states.len()
                } else {
                    control.decisions.is_empty()
                },
                "controlled gate snapshot immediate controller decision receipts are incomplete or unexpected",
            )?;
            let processed = batch
                .controls
                .schedule()
                .decision_times_s
                .partition_point(|time| *time <= control.start_s);
            for (index, receipt) in actuator.receipts.iter().enumerate() {
                let before = if control.decision_index.is_some() {
                    control
                        .decisions
                        .get(index)
                        .map(|decision| &decision.before)
                } else {
                    data.controls.states.get(index)
                }
                .ok_or_else(|| {
                    StructureError::input(
                        StructureInputErrorKind::Definition,
                        "controlled gate snapshot immediate controller before-state is missing",
                    )
                })?;
                super::state::pair(batch, index, before, &receipt.before, processed)?;
                if control.decision_index.is_some() {
                    let after=control.decisions.get(index).map(|decision|&decision.after).ok_or_else(||StructureError::input(StructureInputErrorKind::Definition,"controlled gate snapshot immediate controller after-state is missing"))?;
                    let target = super::state::selected(batch, index, after.latch)?;
                    let update=receipt.update.as_ref().ok_or_else(||StructureError::input(StructureInputErrorKind::Definition,"controlled gate snapshot decision interval lacks its actual endpoint target packet"))?;
                    super::state::packet(batch, index, update, target, control.end_s)?;
                } else {
                    lane(
                        batch,
                        index,
                        receipt.update.is_none(),
                        "controlled gate snapshot nondecision interval contains an endpoint target packet",
                    )?;
                }
            }
            Ok(())
        }
        _ => Err(StructureError::input(
            StructureInputErrorKind::Definition,
            "controlled gate snapshot children disagree on retained immediate interval presence",
        )),
    }
}
