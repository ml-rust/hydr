//! Simultaneous sampled threshold decisions with anchored qualification timers.

use crate::cpu::{
    CpuTensor,
    structures::control::{
        CpuControlContext, CpuControlSample, CpuControlState, CpuControlStep, CpuControlTransition,
        MAX_EXACT_COUNT,
    },
};
use crate::structures::{ControlSample, StructureError};
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps};

pub(crate) fn qualifies(
    context: &CpuControlContext,
    high: &CpuTensor,
    stage: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    let c = &context.structures.client;
    Ok(c.where_cond(
        high,
        &c.le(stage, &context.lower)?,
        &c.ge(stage, &context.upper)?,
    )?)
}
pub(crate) fn decide(
    context: &CpuControlContext,
    before: &CpuControlState,
    start: f64,
    end: f64,
    index: usize,
    samples: &[ControlSample],
    input: CpuControlSample,
) -> Result<CpuControlStep, StructureError> {
    let c = &context.structures.client;
    let qualifies = qualifies(context, &before.high, &input.stage)?;
    let since = c.where_cond(&before.pending, &before.since, &input.observed)?;
    let delay = c.where_cond(&before.high, &context.low_delay, &context.high_delay)?;
    let timer =
        super::timers::evaluate(context, &qualifies, &since, &input.observed, &delay, true)?;
    let transitioned = timer.ready.clone();
    let high = c.where_cond(
        &transitioned,
        &c.sub(&context.one, &before.high)?,
        &before.high,
    )?;
    let pending = c.mul(&qualifies, &c.sub(&context.one, &transitioned)?)?;
    let since = c.where_cond(&pending, &since, &context.zero)?;
    context.reject(
        &c.mul(
            &transitioned,
            &c.ge(
                &before.count,
                &c.mul_scalar(&context.one, MAX_EXACT_COUNT as f64)?,
            )?,
        )?,
        None,
        "control transition counter cannot advance beyond exact integer capacity",
    )?;
    let count = c.add(&before.count, &transitioned)?;
    context.reject(
        &c.gt(&count, &c.mul_scalar(&context.one, MAX_EXACT_COUNT as f64)?)?,
        None,
        "control transition counter exceeds exact integer capacity",
    )?;
    let last_transition = CpuControlTransition {
        valid: c.maximum(&before.last_transition.valid, &transitioned)?,
        decision: c.where_cond(
            &transitioned,
            &c.mul_scalar(&context.one, index as f64)?,
            &before.last_transition.decision,
        )?,
        from: c.where_cond(&transitioned, &before.high, &before.last_transition.from)?,
        sample: super::selection::sample(
            context,
            &transitioned,
            &input,
            &before.last_transition.sample,
        )?,
        timer: super::selection::timer(
            context,
            &transitioned,
            &timer,
            &before.last_transition.timer,
        )?,
    };
    let after = CpuControlState {
        high,
        pending,
        since,
        sample: input,
        transition_valid: c.maximum(&before.transition_valid, &transitioned)?,
        transition_time: c.where_cond(&transitioned, &timer.evaluated, &before.transition_time)?,
        count,
        last_transition,
    };
    Ok(CpuControlStep {
        start,
        end,
        decision_index: Some(index),
        before: before.clone(),
        after,
        samples: samples.to_vec(),
        qualifies,
        timer,
        transitioned,
    })
}
