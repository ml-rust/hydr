//! Exact immutable sampled-control identity shared by snapshots and response models.

use super::{
    ControlCommand, ControlDecisionSchedule, ControlInitialState, ControlSample,
    OperatingControlDefinition, SampledControlTiming,
};
use crate::structures::{
    StructureElement,
    identity::{bits, same_elements},
};
use crate::tensor::structures::control::TensorControlContext;
use numr::dtype::DType;
use numr::runtime::{Device, Runtime};
use std::sync::Arc;

pub(crate) fn same_original<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    elements: &[StructureElement],
    definitions: &[OperatingControlDefinition],
    initial: &[ControlInitialState],
    schedule: &ControlDecisionSchedule,
    timing: &SampledControlTiming,
) -> bool {
    same_elements(&context.structures.elements, elements)
        && same_inputs(context, definitions, initial, schedule, timing)
}
fn same_inputs<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    definitions: &[OperatingControlDefinition],
    initial_states: &[ControlInitialState],
    schedule: &ControlDecisionSchedule,
    timing: &SampledControlTiming,
) -> bool {
    context.definitions.len() == definitions.len()
        && context
            .definitions
            .iter()
            .zip(definitions)
            .all(|(a, b)| definition(a, b))
        && context.initial_states.len() == initial_states.len()
        && context
            .initial_states
            .iter()
            .zip(initial_states)
            .all(|(a, b)| initial(a, b))
        && context.schedule == *schedule
        && bits(
            &[context.schedule.initial_time_s, context.schedule.end_s],
            &[schedule.initial_time_s, schedule.end_s],
        )
        && bits(
            &context.schedule.decision_times_s,
            &schedule.decision_times_s,
        )
        && context.timing == *timing
        && bits(
            &[
                context.timing.maximum_decision_gap_s,
                context.timing.maximum_timer_lateness_s,
            ],
            &[
                timing.maximum_decision_gap_s,
                timing.maximum_timer_lateness_s,
            ],
        )
}
pub(crate) fn same_context<R: Runtime<DType = DType>>(
    a: &Arc<TensorControlContext<R>>,
    b: &Arc<TensorControlContext<R>>,
) -> bool {
    a.structures.device.is_same(&b.structures.device)
        && (Arc::ptr_eq(a, b)
            || ((Arc::ptr_eq(&a.structures, &b.structures)
                || same_elements(&a.structures.elements, &b.structures.elements))
                && same_inputs(a, &b.definitions, &b.initial_states, &b.schedule, &b.timing)))
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
pub(crate) fn command(a: ControlCommand, b: ControlCommand) -> bool {
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
pub(crate) fn sample(a: &ControlSample, b: &ControlSample) -> bool {
    a == b
        && bits(
            &[a.observed_at_s, a.available_at_s, a.stage_m],
            &[b.observed_at_s, b.available_at_s, b.stage_m],
        )
}
pub(crate) fn option_time(a: Option<f64>, b: Option<f64>) -> bool {
    a.map(f64::to_bits) == b.map(f64::to_bits)
}
