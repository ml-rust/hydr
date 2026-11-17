//! Recompute retained transition timers from original operands without history replay.

use super::super::{ControlLatch, ControlState, OperatingControlBatch};
use super::compatibility::require;
use super::state_check::as_definition;
use crate::structures::StructureError;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;

pub(super) fn check<R: Runtime<DType = DType>>(
    batch: &OperatingControlBatch<R>,
    states: &[ControlState],
) -> Result<(), StructureError>
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
    let context = &batch.context;
    let mut active = Vec::with_capacity(states.len());
    let mut starts = Vec::with_capacity(states.len());
    let mut at = Vec::with_capacity(states.len());
    let mut delays = Vec::with_capacity(states.len());
    let mut high = Vec::with_capacity(states.len());
    let mut stages = Vec::with_capacity(states.len());
    for (lane, (state, definition)) in states.iter().zip(batch.definitions()).enumerate() {
        if let Some(receipt) = &state.last_transition {
            let result = (|| {
                crate::structures::tensor::control::input::check_sample(
                    definition,
                    &receipt.sample,
                )?;
                require(
                    receipt.sample.observed_at_s.to_bits() == receipt.time_s.to_bits()
                        && receipt.timer.evaluated_at_s.to_bits() == receipt.time_s.to_bits(),
                    "retained transition sample/timer time differs from original decision",
                )?;
                let delay = if receipt.from == ControlLatch::High {
                    definition.low_qualification_delay_s
                } else {
                    definition.high_qualification_delay_s
                };
                require(
                    receipt.timer.delay_s.to_bits() == delay.to_bits()
                        && receipt.timer.ready
                        && receipt.timer.within_lateness,
                    "retained transition delay, readiness, or lateness differs from original policy",
                )?;
                let initial = &batch.initial_states()[lane];
                let initial_anchor = state.transition_count == 1
                    && receipt.from == initial.latch
                    && initial.pending.is_some_and(|pending| {
                        pending.since_s.to_bits() == receipt.timer.started_at_s.to_bits()
                    });
                let original_anchor = super::compatibility::contains_original_time(
                    &batch.schedule().decision_times_s[..=receipt.decision_index],
                    receipt.timer.started_at_s,
                );
                require(
                    initial_anchor || original_anchor,
                    "retained transition start is not an original decision or initial pending anchor",
                )?;
                require(
                    state.transition_count <= receipt.decision_index + 1,
                    "retained transition predates its claimed in-run count",
                )?;
                if delay == 0.0 {
                    require(
                        receipt.timer.started_at_s.to_bits() == receipt.time_s.to_bits(),
                        "zero-delay transition retains an earlier pending anchor",
                    )?;
                }
                Ok(delay)
            })();
            let delay = result
                .map_err(|error: StructureError| as_definition(error).at(context.slots[lane]))?;
            active.push(1.0);
            starts.push(receipt.timer.started_at_s);
            at.push(receipt.time_s);
            delays.push(delay);
            high.push(if receipt.from == ControlLatch::High {
                1.0
            } else {
                0.0
            });
            stages.push(receipt.sample.stage_m);
        } else {
            active.push(0.0);
            starts.push(0.0);
            at.push(0.0);
            delays.push(0.0);
            high.push(0.0);
            stages.push(0.0);
        }
    }
    let active = context.tensor(&active)?;
    let c = &context.structures.client;
    let qualifies = c.where_cond(
        &context.tensor(&high)?,
        &c.le(&context.tensor(&stages)?, &context.lower)?,
        &c.ge(&context.tensor(&stages)?, &context.upper)?,
    )?;
    context.reject(
        &c.mul(&active, &c.sub(&context.one, &qualifies)?)?,
        Some(crate::structures::StructureInputErrorKind::Definition),
        "retained transition sample does not qualify its prior latch",
    )?;
    let timer = crate::structures::numerics::control::timers::evaluate(
        context,
        &active,
        &context.tensor(&starts)?,
        &context.tensor(&at)?,
        &context.tensor(&delays)?,
        true,
    )
    .map_err(as_definition)?;
    let values = super::super::projection_values::TimerValues::new(context, &timer)?;
    for (lane, state) in states.iter().enumerate() {
        if let Some(receipt) = &state.last_transition {
            require(
                values
                    .at(lane)
                    .as_ref()
                    .is_some_and(|expected| super::receipt_bits::timer(expected, &receipt.timer)),
                "retained transition timer differs from Numr original-operand arithmetic",
            )
            .map_err(|error| error.at(context.slots[lane]))?;
        }
    }
    Ok(())
}
