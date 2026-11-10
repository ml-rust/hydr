//! Exact immutable actuator identity shared by snapshots and source frames.

use crate::structures::identity::{bits, same_elements};
use crate::structures::{
    GateActuatorDefinition, GateActuatorInitialState, GateActuatorSchedule, StructureElement,
};
use crate::tensor::structures::actuator::TensorActuatorContext;
use numr::dtype::DType;
use numr::runtime::{Device, Runtime};
use std::sync::Arc;

pub(crate) fn same_original<R: Runtime<DType = DType>>(
    context: &TensorActuatorContext<R>,
    elements: &[StructureElement],
    definitions: &[GateActuatorDefinition],
    initial: &[GateActuatorInitialState],
    schedule: &GateActuatorSchedule,
) -> bool {
    same_elements(&context.structures.elements, elements)
        && same_inputs(context, definitions, initial, schedule)
}
fn same_inputs<R: Runtime<DType = DType>>(
    context: &TensorActuatorContext<R>,
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
pub(crate) fn same_context<R: Runtime<DType = DType>>(
    a: &Arc<TensorActuatorContext<R>>,
    b: &Arc<TensorActuatorContext<R>>,
) -> bool {
    a.structures.device.is_same(&b.structures.device)
        && (Arc::ptr_eq(a, b)
            || ((Arc::ptr_eq(&a.structures, &b.structures)
                || same_elements(&a.structures.elements, &b.structures.elements))
                && same_inputs(a, &b.definitions, &b.initial_states, &b.schedule)))
}
