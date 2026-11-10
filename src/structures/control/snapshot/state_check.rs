//! Imported causal state, exact counters, and bounded transition consistency.

use super::super::{ControlLatch, ControlState, OperatingControlBatch};
use super::compatibility::{option_time, require, sample};
use crate::structures::{StructureError, StructureInputErrorKind};
use crate::tensor::structures::control::{MAX_EXACT_COUNT, TensorControlState};
use numr::dtype::DType;
use numr::runtime::Runtime;

pub(super) fn check<R: Runtime<DType = DType>>(
    batch: &OperatingControlBatch<R>,
    states: &[ControlState],
    time: f64,
    processed: usize,
) -> Result<TensorControlState<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    require(
        states.len() == batch.definitions().len(),
        "imported control state count differs from ordered definitions",
    )?;
    for (lane, ((definition, state), initial)) in batch
        .definitions()
        .iter()
        .zip(states)
        .zip(batch.initial_states())
        .enumerate()
    {
        let result = (|| {
            require(
                state.controller_id == definition.controller_id,
                "imported controller state identity differs",
            )?;
            require(
                state.transition_count <= processed
                    && state.transition_count as u128 <= MAX_EXACT_COUNT as u128,
                "control transition count exceeds processed decisions or exact integer capacity",
            )?;
            let parity = if state.transition_count % 2 == 0 {
                initial.latch
            } else {
                opposite(initial.latch)
            };
            require(
                state.latch == parity,
                "imported latch disagrees with initial latch and transition-count parity",
            )?;
            if let Some(sample) = &state.last_sample {
                crate::tensor::structures::control::input::check_sample(definition, sample)?;
                require(
                    sample.observed_at_s <= time,
                    "imported control observation follows accepted time",
                )?;
                if processed > 0 {
                    require(
                        sample.observed_at_s.to_bits()
                            == batch.schedule().decision_times_s[processed - 1].to_bits(),
                        "imported last sample does not match last processed original decision",
                    )?;
                } else {
                    require(
                        initial
                            .antecedent_sample
                            .as_ref()
                            .is_some_and(|original| super::compatibility::sample(original, sample)),
                        "unprocessed control state changes its original antecedent sample",
                    )?;
                }
            } else {
                require(
                    processed == 0
                        && initial.antecedent_sample.is_none()
                        && state.pending.is_none(),
                    "imported control state lacks its required last sample",
                )?;
            }
            if let Some(transition) = state.last_transition_at_s {
                require(
                    transition.is_finite()
                        && transition <= time
                        && state
                            .last_sample
                            .as_ref()
                            .is_none_or(|sample| transition <= sample.observed_at_s),
                    "imported transition chronology is inconsistent",
                )?;
            }
            if let Some(pending) = state.pending {
                let observed = state.last_sample.as_ref().ok_or_else(|| {
                    StructureError::input(
                        StructureInputErrorKind::Definition,
                        "imported pending qualification lacks a causal sample",
                    )
                })?;
                require(
                    pending.target == opposite(state.latch)
                        && pending.since_s.is_finite()
                        && pending.since_s <= observed.observed_at_s
                        && state
                            .last_transition_at_s
                            .is_none_or(|transition| pending.since_s > transition),
                    "imported pending target or chronology is inconsistent",
                )?;
                let declared_initial = state.transition_count == 0
                    && initial.pending.is_some_and(|original| {
                        original.since_s.to_bits() == pending.since_s.to_bits()
                    });
                require(
                    declared_initial
                        || super::compatibility::contains_original_time(
                            &batch.schedule().decision_times_s[..processed],
                            pending.since_s,
                        ),
                    "imported pending start is neither an original decision nor retained initial anchor",
                )?;
            }
            if state.transition_count == 0 {
                require(
                    state.last_transition.is_none()
                        && option_time(state.last_transition_at_s, initial.last_transition_at_s),
                    "zero-count control state changes its declared transition history",
                )?;
                if processed == 0 {
                    require(
                        state.pending == initial.pending
                            && state.pending.map(|p| p.since_s.to_bits())
                                == initial.pending.map(|p| p.since_s.to_bits())
                            && match (&state.last_sample, &initial.antecedent_sample) {
                                (Some(a), Some(b)) => sample(a, b),
                                (None, None) => true,
                                _ => false,
                            },
                        "initial imported relay memory differs from explicit initial state",
                    )?;
                }
            } else {
                let receipt = state.last_transition.as_ref().ok_or_else(|| {
                    StructureError::input(
                        StructureInputErrorKind::Definition,
                        "positive transition count lacks complete last transition",
                    )
                })?;
                require(
                    receipt.decision_index < processed
                        && receipt.to == state.latch
                        && receipt.from == opposite(state.latch)
                        && receipt.time_s.to_bits()
                            == batch.schedule().decision_times_s[receipt.decision_index].to_bits()
                        && option_time(Some(receipt.time_s), state.last_transition_at_s),
                    "imported last transition disagrees with state or original schedule",
                )?;
                if receipt.decision_index == processed - 1 {
                    require(
                        state
                            .last_sample
                            .as_ref()
                            .is_some_and(|latest| sample(latest, &receipt.sample)),
                        "latest retained transition and current state contain different causal sample bits",
                    )?;
                }
            }
            Ok(())
        })();
        result
            .map_err(|error: StructureError| as_definition(error).at(batch.context.slots[lane]))?;
    }
    let state = TensorControlState::from_values(&batch.context, states)?;
    crate::numerics::structures::control::check_initial(&batch.context, &state)
        .map_err(as_definition)?;
    super::transition_check::check(batch, states)?;
    Ok(state)
}
pub(super) fn as_definition(error: StructureError) -> StructureError {
    if matches!(error, StructureError::Runtime { .. }) {
        return error;
    }
    let index = error.element();
    let error = StructureError::input(
        StructureInputErrorKind::Definition,
        format!("imported control state is inconsistent: {error}"),
    );
    match index {
        Some(index) => error.at(index),
        None => error,
    }
}
fn opposite(latch: ControlLatch) -> ControlLatch {
    match latch {
        ControlLatch::Low => ControlLatch::High,
        ControlLatch::High => ControlLatch::Low,
    }
}
