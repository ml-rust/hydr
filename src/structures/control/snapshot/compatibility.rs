//! Exact complete original definition compatibility, including signed zero.

use super::super::{
    ControlCommand, ControlInitialState, ControlSample, OperatingControlBatch,
    OperatingControlDefinition,
};
use crate::structures::identity::bits;
use crate::structures::{StructureError, StructureInputErrorKind};

pub(super) fn require(condition: bool, message: &str) -> Result<(), StructureError> {
    if !condition {
        return Err(StructureError::input(
            StructureInputErrorKind::Definition,
            message,
        ));
    }
    Ok(())
}
pub(super) fn check(
    batch: &OperatingControlBatch,
    data: &super::OperatingControlSnapshotData,
) -> Result<(), StructureError> {
    require(
        crate::structures::identity::same_elements(batch.elements(), &data.elements)
            && batch.definitions().len() == data.definitions.len()
            && batch
                .definitions()
                .iter()
                .zip(&data.definitions)
                .all(|(a, b)| definition(a, b))
            && batch.initial_states().len() == data.initial_states.len()
            && batch
                .initial_states()
                .iter()
                .zip(&data.initial_states)
                .all(|(a, b)| initial(a, b))
            && batch.schedule() == &data.schedule
            && bits(
                &[batch.schedule().initial_time_s, batch.schedule().end_s],
                &[data.schedule.initial_time_s, data.schedule.end_s],
            )
            && bits(
                &batch.schedule().decision_times_s,
                &data.schedule.decision_times_s,
            )
            && batch.timing() == &data.timing
            && bits(
                &[
                    batch.timing().maximum_decision_gap_s,
                    batch.timing().maximum_timer_lateness_s,
                ],
                &[
                    data.timing.maximum_decision_gap_s,
                    data.timing.maximum_timer_lateness_s,
                ],
            ),
        "control snapshot definitions differ: restore into an identically constructed actual structure/control batch",
    )
}
fn definition(a: &OperatingControlDefinition, b: &OperatingControlDefinition) -> bool {
    a == b
        && bits(
            &[
                a.lower_stage_m,
                a.upper_stage_m,
                a.high_qualification_delay_s,
                a.low_qualification_delay_s,
            ],
            &[
                b.lower_stage_m,
                b.upper_stage_m,
                b.high_qualification_delay_s,
                b.low_qualification_delay_s,
            ],
        )
        && command(a.low_command, b.low_command)
        && command(a.high_command, b.high_command)
}
pub(super) fn command(a: ControlCommand, b: ControlCommand) -> bool {
    match (a, b) {
        (ControlCommand::PumpStoppedIsolated, ControlCommand::PumpStoppedIsolated) => true,
        (
            ControlCommand::PumpRunning { speed_ratio: a },
            ControlCommand::PumpRunning { speed_ratio: b },
        ) => a.to_bits() == b.to_bits(),
        (
            ControlCommand::GateTarget { opening_m: a },
            ControlCommand::GateTarget { opening_m: b },
        ) => a.to_bits() == b.to_bits(),
        _ => false,
    }
}
fn initial(a: &ControlInitialState, b: &ControlInitialState) -> bool {
    a == b
        && match (a.pending, b.pending) {
            (Some(a), Some(b)) => a.since_s.to_bits() == b.since_s.to_bits(),
            (None, None) => true,
            _ => false,
        }
        && option_time(a.last_transition_at_s, b.last_transition_at_s)
        && match (&a.antecedent_sample, &b.antecedent_sample) {
            (Some(a), Some(b)) => sample(a, b),
            (None, None) => true,
            _ => false,
        }
}
pub(super) fn sample(a: &ControlSample, b: &ControlSample) -> bool {
    a == b
        && bits(
            &[a.observed_at_s, a.available_at_s, a.stage_m],
            &[b.observed_at_s, b.available_at_s, b.stage_m],
        )
}
pub(super) fn option_time(a: Option<f64>, b: Option<f64>) -> bool {
    a.map(f64::to_bits) == b.map(f64::to_bits)
}

/// Checked schedules increase strictly. Total ordering preserves original signed-zero identity.
pub(super) fn contains_original_time(times: &[f64], time: f64) -> bool {
    times
        .binary_search_by(|candidate| candidate.total_cmp(&time))
        .is_ok()
}

pub(super) fn check_batch(
    a: &OperatingControlBatch,
    b: &OperatingControlBatch,
) -> Result<(), StructureError> {
    if std::sync::Arc::ptr_eq(&a.context, &b.context) {
        return Ok(());
    }
    require(
        crate::structures::identity::same_elements(a.elements(), b.elements())
            && a.definitions().len() == b.definitions().len()
            && a.definitions()
                .iter()
                .zip(b.definitions())
                .all(|(a, b)| definition(a, b))
            && a.initial_states().len() == b.initial_states().len()
            && a.initial_states()
                .iter()
                .zip(b.initial_states())
                .all(|(a, b)| initial(a, b))
            && a.schedule() == b.schedule()
            && bits(
                &[a.schedule().initial_time_s, a.schedule().end_s],
                &[b.schedule().initial_time_s, b.schedule().end_s],
            )
            && bits(
                &a.schedule().decision_times_s,
                &b.schedule().decision_times_s,
            )
            && a.timing() == b.timing()
            && bits(
                &[
                    a.timing().maximum_decision_gap_s,
                    a.timing().maximum_timer_lateness_s,
                ],
                &[
                    b.timing().maximum_decision_gap_s,
                    b.timing().maximum_timer_lateness_s,
                ],
            ),
        "trusted control snapshot definitions differ from destination context",
    )
}
