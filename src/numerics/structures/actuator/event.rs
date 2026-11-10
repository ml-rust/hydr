//! Scalar event reductions over complete batched original arrival certificates.

use super::{arithmetic::not, evaluate::current_completion};
use crate::structures::{GateActuatorEvent, StructureError};
use crate::tensor::structures::actuator::{TensorActuatorContext, TensorActuatorState};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps};
use numr::runtime::Runtime;

pub(crate) fn next_event<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    state: &TensorActuatorState<R>,
    time: f64,
    next_target: usize,
) -> Result<Option<GateActuatorEvent>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>,
{
    if time == ctx.schedule.end_s {
        return Ok(None);
    }
    let c = &ctx.structures.client;
    let now = ctx.constant(time)?;
    let terminal = ctx.constant(ctx.schedule.end_s)?;
    let command = ctx.schedule.target_times_s.get(next_target).copied();
    let fallback = command
        .unwrap_or(ctx.schedule.end_s)
        .min(ctx.schedule.end_s);
    let finite = c.eq(&state.arrival.kind, &ctx.one)?;
    let pending = c.mul(&finite, &not(ctx, &current_completion(ctx, state)?)?)?;
    let lower = c.mul(
        &pending,
        &c.mul(
            &c.gt(&state.arrival.lower, &now)?,
            &c.le(&state.arrival.lower, &terminal)?,
        )?,
    )?;
    let upper = c.mul(
        &pending,
        &c.mul(
            &c.gt(&state.arrival.upper, &now)?,
            &c.le(&state.arrival.upper, &terminal)?,
        )?,
    )?;
    let base = ctx.constant(fallback)?;
    let candidates = c.minimum(
        &base,
        &c.minimum(
            &c.where_cond(&lower, &state.arrival.lower, &base)?,
            &c.where_cond(&upper, &state.arrival.upper, &base)?,
        )?,
    )?;
    let time_s = c.min(&candidates, &[], false)?.item::<f64>()?;
    if !time_s.is_finite() || time_s <= time {
        return Err(StructureError::numerical(
            "actuator next event does not strictly advance its accepted clock",
        ));
    }
    let selected = ctx.constant(time_s)?;
    let arrival_lower_boundary = ctx
        .structures
        .any(&c.mul(&lower, &c.eq(&state.arrival.lower, &selected)?)?)?;
    let arrival_upper_boundary = ctx
        .structures
        .any(&c.mul(&upper, &c.eq(&state.arrival.upper, &selected)?)?)?;
    Ok(Some(GateActuatorEvent {
        time_s,
        target_index: if command == Some(time_s) {
            Some(next_target)
        } else {
            None
        },
        arrival_lower_boundary,
        arrival_upper_boundary,
        terminal: time_s == ctx.schedule.end_s,
    }))
}
