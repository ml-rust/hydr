//! Shared analytic or forward completion authority without repeated arrival counting.

use super::{
    arithmetic::{not, used, width},
    forward::forward,
};
use crate::structures::StructureError;
use crate::structures::tensor::actuator::{
    TensorActuatorArrival, TensorActuatorCompletion, TensorActuatorContext, TensorActuatorForward,
    TensorActuatorPosition, TensorActuatorState, record::TensorRecord,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) struct TensorActuatorEvaluation<R: Runtime<DType = DType>> {
    pub position: TensorActuatorPosition<R>,
    pub forward: TensorActuatorForward<R>,
    pub completion: Arc<TensorActuatorCompletion<R>>,
    pub completed: Tensor<R>,
    pub phase: Tensor<R>,
}
pub(crate) fn current_completion<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    state: &TensorActuatorState<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::CompareOps<R>,
{
    let c = &ctx.structures.client;
    Ok(c.mul(
        &state.active_valid,
        &c.mul(
            &state.last_completion.valid,
            &c.mul(
                &c.eq(&state.last_completion.generation, &state.generation)?,
                &c.eq(&state.last_completion.episode, &state.active_index)?,
            )?,
        )?,
    )?)
}
pub(crate) fn phase<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    arrival: &TensorActuatorArrival<R>,
    time: f64,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::CompareOps<R> + numr::ops::ConditionalOps<R>,
{
    phase_at(ctx, arrival, &ctx.constant(time)?)
}
pub(crate) fn phase_at<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    arrival: &TensorActuatorArrival<R>,
    t: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::CompareOps<R> + numr::ops::ConditionalOps<R>,
{
    let c = &ctx.structures.client;
    let finite = c.eq(&arrival.kind, &ctx.one)?;
    let hold = c.eq(&arrival.kind, &ctx.zero)?;
    let finite_phase = c.where_cond(
        &c.lt(t, &arrival.lower)?,
        &ctx.zero,
        &c.where_cond(&c.lt(t, &arrival.upper)?, &ctx.one, &ctx.constant(2.0)?)?,
    )?;
    let absent = c.where_cond(
        &c.eq(&arrival.kind, &ctx.constant(2.0)?)?,
        &ctx.constant(3.0)?,
        &c.where_cond(
            &c.eq(&arrival.kind, &ctx.constant(3.0)?)?,
            &ctx.constant(4.0)?,
            &ctx.constant(5.0)?,
        )?,
    )?;
    Ok(c.where_cond(
        &finite,
        &finite_phase,
        &c.where_cond(&hold, &ctx.constant(2.0)?, &absent)?,
    )?)
}
pub(crate) fn evaluate<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    state: &TensorActuatorState<R>,
    query: f64,
) -> Result<TensorActuatorEvaluation<R>, StructureError>
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
    evaluate_at(ctx, state, &ctx.constant(query)?)
}
pub(crate) fn evaluate_at<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    state: &TensorActuatorState<R>,
    t: &Tensor<R>,
) -> Result<TensorActuatorEvaluation<R>, StructureError>
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
    let c = &ctx.structures.client;
    let authority = current_completion(ctx, state)?;
    let finite = c.eq(&state.arrival.kind, &ctx.one)?;
    let analytic = c.mul(&finite, &c.ge(t, &state.arrival.upper)?)?;
    let hold = c.eq(&state.arrival.kind, &ctx.zero)?;
    let known = c.maximum(&hold, &c.maximum(&authority, &analytic)?)?;
    let forward_active = not(ctx, &known)?;
    let forward = forward(ctx, &state.anchor, &state.target, t, &forward_active)?;
    let singleton = c.mul(
        &forward_active,
        &c.mul(
            &c.eq(&forward.position.lower, &state.target)?,
            &c.eq(&forward.position.upper, &state.target)?,
        )?,
    )?;
    let proof = c.maximum(&analytic, &singleton)?;
    let completed = c.mul(&state.active_valid, &c.mul(&not(ctx, &authority)?, &proof)?)?;
    used(
        ctx,
        &completed,
        &finite,
        "actuator forward completion contradicts retained arrival support",
    )?;
    used(
        ctx,
        &completed,
        &c.mul(
            &c.gt(t, &state.anchor.epoch)?,
            &c.ge(t, &state.arrival.lower)?,
        )?,
        "actuator completion contradicts activation or certified lower time",
    )?;
    let effective_upper = c.minimum(&state.arrival.upper, t)?;
    let (wh, wl) = width(ctx, &state.arrival.lower, &effective_upper, &completed)?;
    let receipt_forward = c.mul(&completed, &c.mul(&singleton, &not(ctx, &analytic)?)?)?;
    let candidate = TensorActuatorCompletion {
        valid: completed.clone(),
        episode: state.active_index.clone(),
        generation: state.generation.clone(),
        confirmed: t.clone(),
        anchor: state.anchor.clone(),
        target: state.target.clone(),
        arrival: state.arrival.clone(),
        effective_upper,
        width_high: wh,
        width_low: wl,
        forward: Arc::new(TensorActuatorForward::select(
            ctx,
            &receipt_forward,
            &forward,
            &TensorActuatorForward::zero(ctx),
        )?),
    };
    let completion = Arc::new(TensorActuatorCompletion::select(
        ctx,
        &completed,
        &candidate,
        &state.last_completion,
    )?);
    let complete = c.maximum(&known, &singleton)?;
    let canonical = TensorActuatorPosition {
        center: state.target.clone(),
        lower: state.target.clone(),
        upper: state.target.clone(),
        error: ctx.zero.clone(),
    };
    let position = TensorActuatorPosition::select(ctx, &complete, &canonical, &forward.position)?;
    Ok(TensorActuatorEvaluation {
        position,
        forward,
        completion,
        completed,
        phase: c.where_cond(
            &complete,
            &ctx.constant(2.0)?,
            &phase_at(ctx, &state.arrival, t)?,
        )?,
    })
}
