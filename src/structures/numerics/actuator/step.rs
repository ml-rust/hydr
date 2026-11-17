//! Ordered old-target left evaluation and causal endpoint target installation.

use super::{
    arithmetic::{count, not},
    arrival::arrival,
    evaluate::{current_completion, evaluate, phase},
};
use crate::structures::tensor::actuator::{
    TensorActuatorAnchor, TensorActuatorArrival, TensorActuatorChange, TensorActuatorCompletion,
    TensorActuatorContext, TensorActuatorPosition, TensorActuatorState, TensorActuatorStep,
    TensorActuatorUpdate, record::TensorRecord,
};
use crate::structures::{GateActuatorEvent, StructureError};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use std::sync::Arc;

pub(crate) fn step<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    before: &TensorActuatorState<R>,
    start: f64,
    end: f64,
    event: Option<GateActuatorEvent>,
    target_index: Option<usize>,
    update: TensorActuatorUpdate<R>,
) -> Result<TensorActuatorStep<R>, StructureError>
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
    let evaluation = evaluate(ctx, before, end)?;
    let changed = c.mul(&update.valid, &c.ne(&update.target, &before.target)?)?;
    let mut left = before.clone();
    left.position = evaluation.position.clone();
    left.phase = evaluation.phase.clone();
    left.last_completion = evaluation.completion.clone();
    left.arrivals = count(ctx, &before.arrivals, &evaluation.completed)?;
    let new_hold = c.mul(
        &changed,
        &c.mul(
            &c.eq(&left.position.lower, &update.target)?,
            &c.eq(&left.position.upper, &update.target)?,
        )?,
    )?;
    let canonical = TensorActuatorPosition {
        center: update.target.clone(),
        lower: update.target.clone(),
        upper: update.target.clone(),
        error: ctx.zero.clone(),
    };
    let new_position = TensorActuatorPosition::select(ctx, &new_hold, &canonical, &left.position)?;
    let new_anchor = TensorActuatorAnchor {
        epoch: ctx.constant(end)?,
        position: new_position.clone(),
    };
    let masked_anchor = TensorActuatorAnchor::select(
        ctx,
        &changed,
        &new_anchor,
        &TensorActuatorAnchor {
            epoch: ctx.constant(end)?,
            position: TensorActuatorPosition::zero(ctx),
        },
    )?;
    let masked_target = c.where_cond(&changed, &update.target, &ctx.zero)?;
    let next_arrival = arrival(ctx, &masked_anchor, &masked_target)?;
    let new_active = c.mul(&changed, &not(ctx, &new_hold)?)?;
    let episodes = count(ctx, &before.episodes, &new_active)?;
    let generation = count(ctx, &before.generation, &changed)?;
    let new_index = c.where_cond(&new_active, &c.sub(&episodes, &ctx.one)?, &ctx.zero)?;
    let previous_authority = current_completion(ctx, &left)?;
    let receipt = TensorActuatorChange {
        valid: changed.clone(),
        index: ctx.constant(target_index.unwrap_or(0) as f64)?,
        time: ctx.constant(end)?,
        previous_target: before.target.clone(),
        previous_anchor: before.anchor.clone(),
        previous_completion: TensorActuatorCompletion::select(
            ctx,
            &previous_authority,
            &left.last_completion,
            &TensorActuatorCompletion::zero(ctx),
        )?,
        update: update.clone(),
        left: left.position.clone(),
        new_anchor: new_anchor.clone(),
        generation: generation.clone(),
        new_episode_valid: new_active.clone(),
        new_episode: new_index.clone(),
    };
    let mut after = left;
    after.anchor = TensorActuatorAnchor::select(ctx, &changed, &new_anchor, &before.anchor)?;
    after.target = c.where_cond(&changed, &update.target, &before.target)?;
    after.position = TensorActuatorPosition::select(ctx, &changed, &new_position, &after.position)?;
    after.phase = c.where_cond(&changed, &phase(ctx, &next_arrival, end)?, &after.phase)?;
    after.arrival = TensorActuatorArrival::select(ctx, &changed, &next_arrival, &before.arrival)?;
    after.generation = generation;
    after.episodes = episodes;
    after.active_valid = c.where_cond(&changed, &new_active, &before.active_valid)?;
    after.active_index = c.where_cond(&changed, &new_index, &before.active_index)?;
    after.latest_update =
        TensorActuatorUpdate::select(ctx, &update.valid, &update, &before.latest_update)?;
    after.last_change = Arc::new(TensorActuatorChange::select(
        ctx,
        &changed,
        &receipt,
        &before.last_change,
    )?);
    Ok(TensorActuatorStep {
        start,
        end,
        event,
        before: before.clone(),
        after,
        left: evaluation.position,
        forward: evaluation.forward,
        update,
        completed: evaluation.completed,
        changed,
    })
}
