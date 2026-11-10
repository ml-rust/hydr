//! Exact immutable actuator identity shared by snapshots and source frames.

use crate::cpu::structures::actuator::CpuActuatorContext;
use crate::structures::identity::{bits, same_elements};
use crate::structures::{
    GateActuatorDefinition, GateActuatorInitialState, GateActuatorSchedule, StructureElement,
};
use std::sync::Arc;

pub(crate) fn same_original(
    context: &CpuActuatorContext,
    elements: &[StructureElement],
    definitions: &[GateActuatorDefinition],
    initial: &[GateActuatorInitialState],
    schedule: &GateActuatorSchedule,
) -> bool {
    same_elements(&context.structures.elements, elements)
        && same_inputs(context, definitions, initial, schedule)
}
fn same_inputs(
    context: &CpuActuatorContext,
    definitions: &[GateActuatorDefinition],
    initial: &[GateActuatorInitialState],
    schedule: &GateActuatorSchedule,
) -> bool {
    context.definitions.len() == definitions.len()
        && context.definitions.iter().zip(definitions).all(|(a, b)| {
            a == b
                && bits(
                    &[
                        a.opening_rate_m_per_s,
                        a.closing_rate_m_per_s,
                        a.accuracy.maximum_absolute_opening_error_m,
                        a.accuracy.maximum_relative_opening_error,
                        a.accuracy.maximum_arrival_window_s,
                    ],
                    &[
                        b.opening_rate_m_per_s,
                        b.closing_rate_m_per_s,
                        b.accuracy.maximum_absolute_opening_error_m,
                        b.accuracy.maximum_relative_opening_error,
                        b.accuracy.maximum_arrival_window_s,
                    ],
                )
        })
        && context.initial_states.len() == initial.len()
        && context.initial_states.iter().zip(initial).all(|(a, b)| {
            a == b
                && bits(
                    &[a.opening_m, a.target_opening_m],
                    &[b.opening_m, b.target_opening_m],
                )
        })
        && context.schedule == *schedule
        && bits(
            &[context.schedule.initial_time_s, context.schedule.end_s],
            &[schedule.initial_time_s, schedule.end_s],
        )
        && bits(&context.schedule.target_times_s, &schedule.target_times_s)
}
pub(crate) fn same_context(a: &Arc<CpuActuatorContext>, b: &Arc<CpuActuatorContext>) -> bool {
    Arc::ptr_eq(a, b)
        || ((Arc::ptr_eq(&a.structures, &b.structures)
            || same_elements(&a.structures.elements, &b.structures.elements))
            && same_inputs(a, &b.definitions, &b.initial_states, &b.schedule))
}
