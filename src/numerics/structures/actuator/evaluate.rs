//! Shared analytic or forward completion authority without repeated arrival counting.

use super::{
    arithmetic::{not, used, width},
    forward::forward,
};
use crate::cpu::{
    CpuTensor,
    structures::actuator::{
        CpuActuatorArrival, CpuActuatorCompletion, CpuActuatorContext, CpuActuatorForward,
        CpuActuatorPosition, CpuActuatorState, record::TensorRecord,
    },
};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};

#[derive(Clone, Debug)]
pub(crate) struct CpuActuatorEvaluation {
    pub position: CpuActuatorPosition,
    pub forward: CpuActuatorForward,
    pub completion: CpuActuatorCompletion,
    pub completed: CpuTensor,
    pub phase: CpuTensor,
}
pub(crate) fn current_completion(
    ctx: &CpuActuatorContext,
    state: &CpuActuatorState,
) -> Result<CpuTensor, StructureError> {
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
pub(crate) fn phase(
    ctx: &CpuActuatorContext,
    arrival: &CpuActuatorArrival,
    time: f64,
) -> Result<CpuTensor, StructureError> {
    phase_at(ctx, arrival, &ctx.constant(time)?)
}
pub(crate) fn phase_at(
    ctx: &CpuActuatorContext,
    arrival: &CpuActuatorArrival,
    t: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
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
pub(crate) fn evaluate(
    ctx: &CpuActuatorContext,
    state: &CpuActuatorState,
    query: f64,
) -> Result<CpuActuatorEvaluation, StructureError> {
    evaluate_at(ctx, state, &ctx.constant(query)?)
}
pub(crate) fn evaluate_at(
    ctx: &CpuActuatorContext,
    state: &CpuActuatorState,
    t: &CpuTensor,
) -> Result<CpuActuatorEvaluation, StructureError> {
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
    let candidate = CpuActuatorCompletion {
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
        forward: CpuActuatorForward::select(
            ctx,
            &receipt_forward,
            &forward,
            &CpuActuatorForward::zero(ctx),
        )?,
    };
    let completion =
        CpuActuatorCompletion::select(ctx, &completed, &candidate, &state.last_completion)?;
    let complete = c.maximum(&known, &singleton)?;
    let canonical = CpuActuatorPosition {
        center: state.target.clone(),
        lower: state.target.clone(),
        upper: state.target.clone(),
        error: ctx.zero.clone(),
    };
    let position = CpuActuatorPosition::select(ctx, &complete, &canonical, &forward.position)?;
    Ok(CpuActuatorEvaluation {
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
