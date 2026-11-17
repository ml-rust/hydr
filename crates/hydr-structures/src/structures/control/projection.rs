//! Explicit controller value projection without numerical host-state advancement.

use super::projection_values::{SampleValues, TimerValues, index, read};
use super::{
    ControlCommandOutput, ControlDecisionReceipt, ControlLatch, ControlPending, ControlState,
    ControlStepDiagnostics, ControlTransitionReceipt,
};
use crate::structures::StructureError;
use crate::structures::tensor::control::{
    TensorControlContext, TensorControlState, TensorControlStep,
};
use numr::dtype::DType;
use numr::runtime::Runtime;

pub(crate) fn commands<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    state: &TensorControlState<R>,
    time_s: f64,
) -> Result<Vec<ControlCommandOutput>, StructureError> {
    let high = read(context, &state.high)?;
    Ok(high
        .iter()
        .enumerate()
        .map(|(lane, &high)| command(context, lane, high != 0.0, time_s))
        .collect())
}
fn command<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    lane: usize,
    high: bool,
    time_s: f64,
) -> ControlCommandOutput {
    let definition = &context.definitions[lane];
    ControlCommandOutput {
        element: context.slots[lane],
        controller_id: definition.controller_id.clone(),
        structure_id: definition.structure_id.clone(),
        time_s,
        latch: if high {
            ControlLatch::High
        } else {
            ControlLatch::Low
        },
        command: if high {
            definition.high_command
        } else {
            definition.low_command
        },
    }
}
pub(crate) fn states<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    state: &TensorControlState<R>,
) -> Result<Vec<ControlState>, StructureError> {
    let high = read(context, &state.high)?;
    let pending = read(context, &state.pending)?;
    let since = read(context, &state.since)?;
    let transition_valid = read(context, &state.transition_valid)?;
    let transition_time = read(context, &state.transition_time)?;
    let count = read(context, &state.count)?;
    let last_valid = read(context, &state.last_transition.valid)?;
    let last_decision = read(context, &state.last_transition.decision)?;
    let last_from = read(context, &state.last_transition.from)?;
    let sample = SampleValues::new(context, &state.sample)?;
    let last_sample = SampleValues::new(context, &state.last_transition.sample)?;
    let timer = TimerValues::new(context, &state.last_transition.timer)?;
    (0..context.definitions.len())
        .map(|lane| {
            let latch = if high[lane] != 0.0 {
                ControlLatch::High
            } else {
                ControlLatch::Low
            };
            let last_transition = if last_valid[lane] != 0.0 {
                let from = if last_from[lane] != 0.0 {
                    ControlLatch::High
                } else {
                    ControlLatch::Low
                };
                let sample = last_sample.at(context, lane).ok_or_else(|| {
                    StructureError::numerical("retained control transition lacks its sample")
                        .at(context.slots[lane])
                })?;
                let timer = timer.at(lane).ok_or_else(|| {
                    StructureError::numerical("retained control transition lacks its timer")
                        .at(context.slots[lane])
                })?;
                Some(ControlTransitionReceipt {
                    decision_index: index(last_decision[lane])?,
                    time_s: timer.evaluated_at_s,
                    from,
                    to: if from == ControlLatch::High {
                        ControlLatch::Low
                    } else {
                        ControlLatch::High
                    },
                    sample,
                    timer,
                })
            } else {
                None
            };
            Ok(ControlState {
                controller_id: context.definitions[lane].controller_id.clone(),
                latch,
                pending: if pending[lane] != 0.0 {
                    Some(ControlPending {
                        target: if latch == ControlLatch::High {
                            ControlLatch::Low
                        } else {
                            ControlLatch::High
                        },
                        since_s: since[lane],
                    })
                } else {
                    None
                },
                last_sample: sample.at(context, lane),
                last_transition_at_s: if transition_valid[lane] != 0.0 {
                    Some(transition_time[lane])
                } else {
                    None
                },
                transition_count: index(count[lane])?,
                last_transition,
            })
        })
        .collect()
}
pub(crate) fn step<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    step: &TensorControlStep<R>,
) -> Result<ControlStepDiagnostics, StructureError> {
    let before = states(context, &step.before)?;
    let after = states(context, &step.after)?;
    let left_commands = before
        .iter()
        .enumerate()
        .map(|(lane, state)| command(context, lane, state.latch == ControlLatch::High, step.start))
        .collect();
    let right_commands = after
        .iter()
        .enumerate()
        .map(|(lane, state)| command(context, lane, state.latch == ControlLatch::High, step.end))
        .collect();
    let mut decisions = Vec::new();
    if step.decision_index.is_some() {
        let qualifies = read(context, &step.qualifies)?;
        let transitioned = read(context, &step.transitioned)?;
        let timer = TimerValues::new(context, &step.timer)?;
        for lane in 0..context.definitions.len() {
            decisions.push(ControlDecisionReceipt {
                element: context.slots[lane],
                controller_id: context.definitions[lane].controller_id.clone(),
                before: before[lane].clone(),
                sample: step.samples[lane].clone(),
                qualifies: qualifies[lane] != 0.0,
                timer: timer.at(lane),
                transitioned: transitioned[lane] != 0.0,
                after: after[lane].clone(),
            });
        }
    }
    Ok(ControlStepDiagnostics {
        start_s: step.start,
        end_s: step.end,
        decision_index: step.decision_index,
        left_commands,
        right_commands,
        decisions,
    })
}
