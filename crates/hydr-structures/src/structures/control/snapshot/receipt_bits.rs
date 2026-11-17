//! Exact projected receipt equality without treating signed zero as interchangeable.

use super::super::{
    ControlState, ControlStepDiagnostics, ControlTimerWitness, ControlTransitionReceipt,
};
use super::compatibility::{command, option_time, sample};
use crate::structures::identity::bits;

pub(super) fn timer(a: &ControlTimerWitness, b: &ControlTimerWitness) -> bool {
    a == b
        && bits(
            &[
                a.started_at_s,
                a.evaluated_at_s,
                a.delay_s,
                a.elapsed_high_s,
                a.elapsed_low_s,
            ],
            &[
                b.started_at_s,
                b.evaluated_at_s,
                b.delay_s,
                b.elapsed_high_s,
                b.elapsed_low_s,
            ],
        )
        && match (a.allowed_elapsed, b.allowed_elapsed) {
            (Some(a), Some(b)) => bits(&[a.high_s, a.low_s], &[b.high_s, b.low_s]),
            (None, None) => true,
            _ => false,
        }
}
pub(super) fn transition(a: &ControlTransitionReceipt, b: &ControlTransitionReceipt) -> bool {
    a == b
        && a.time_s.to_bits() == b.time_s.to_bits()
        && sample(&a.sample, &b.sample)
        && timer(&a.timer, &b.timer)
}
pub(super) fn state(a: &ControlState, b: &ControlState) -> bool {
    a == b
        && option_time(a.last_transition_at_s, b.last_transition_at_s)
        && match (a.pending, b.pending) {
            (Some(a), Some(b)) => a.since_s.to_bits() == b.since_s.to_bits(),
            (None, None) => true,
            _ => false,
        }
        && match (&a.last_sample, &b.last_sample) {
            (Some(a), Some(b)) => sample(a, b),
            (None, None) => true,
            _ => false,
        }
        && match (&a.last_transition, &b.last_transition) {
            (Some(a), Some(b)) => transition(a, b),
            (None, None) => true,
            _ => false,
        }
}
pub(super) fn states(a: &[ControlState], b: &[ControlState]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| state(a, b))
}
pub(super) fn step(a: &ControlStepDiagnostics, b: &ControlStepDiagnostics) -> bool {
    a == b
        && bits(&[a.start_s, a.end_s], &[b.start_s, b.end_s])
        && a.left_commands
            .iter()
            .zip(&b.left_commands)
            .chain(a.right_commands.iter().zip(&b.right_commands))
            .all(|(a, b)| a.time_s.to_bits() == b.time_s.to_bits() && command(a.command, b.command))
        && a.decisions.iter().zip(&b.decisions).all(|(a, b)| {
            state(&a.before, &b.before)
                && state(&a.after, &b.after)
                && sample(&a.sample, &b.sample)
                && match (&a.timer, &b.timer) {
                    (Some(a), Some(b)) => timer(a, b),
                    (None, None) => true,
                    _ => false,
                }
        })
}
