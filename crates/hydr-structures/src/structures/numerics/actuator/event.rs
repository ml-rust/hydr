//! Scalar event reductions over complete batched original arrival certificates.
//!
//! Each pending finite arrival holds a certified window `[lower, upper]`. The window encloses
//! one arrival instant, and its width comes from clock rounding and the anchor enclosure.
//!
//! - [`next_event`] stops at each bound. A standalone walk ends one interval at each.
//! - [`next_coordinated_event`] counts each window as one instant. A coordinator with a smallest
//!   admitted step then never ends an interval between two bounds of one window. An interval
//!   may end anywhere inside a window ahead, and an interval that starts inside a window may
//!   cross its upper bound. No interval then holds a whole window strictly inside it.

use super::{arithmetic::not, evaluate::current_completion};
use crate::structures::tensor::actuator::{TensorActuatorContext, TensorActuatorState};
use crate::structures::{GateActuatorEvent, StructureError};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Pending arrival bounds strictly after the accepted `time` and at or before the terminal
/// horizon: the lower mask, then the upper mask.
fn bounds<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    state: &TensorActuatorState<R>,
    time: f64,
) -> Result<(Tensor<R>, Tensor<R>), StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::CompareOps<R> + numr::ops::ConditionalOps<R>,
{
    let c = &ctx.structures.client;
    let now = ctx.constant(time)?;
    let terminal = ctx.constant(ctx.schedule.end_s)?;
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
    Ok((lower, upper))
}

/// Target time after `next_target`, and the earlier of it and the terminal horizon.
fn command<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    next_target: usize,
) -> (Option<f64>, f64) {
    let command = ctx.schedule.target_times_s.get(next_target).copied();
    let fallback = command
        .unwrap_or(ctx.schedule.end_s)
        .min(ctx.schedule.end_s);
    (command, fallback)
}

/// Smallest of `fallback` and every time under its mask in `masked`, strictly after `time`.
fn earliest<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    fallback: f64,
    masked: &[(&Tensor<R>, &Tensor<R>)],
    time: f64,
) -> Result<f64, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>,
{
    let c = &ctx.structures.client;
    let base = ctx.constant(fallback)?;
    let mut candidates = base.clone();
    for (mask, times) in masked {
        candidates = c.minimum(&candidates, &c.where_cond(mask, times, &base)?)?;
    }
    let time_s = c.min(&candidates, &[], false)?.item::<f64>()?;
    if !time_s.is_finite() || time_s <= time {
        return Err(StructureError::numerical(
            "actuator next event does not strictly advance its accepted clock",
        ));
    }
    Ok(time_s)
}

/// The event exactly at `at` after the accepted `time`, or `None` when `at` holds no target,
/// pending arrival bound, or terminal horizon.
pub(crate) fn event_at<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    state: &TensorActuatorState<R>,
    (time, next_target): (f64, usize),
    at: f64,
) -> Result<Option<GateActuatorEvent>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>,
{
    let c = &ctx.structures.client;
    let (lower, upper) = bounds(ctx, state, time)?;
    let (command, _) = command(ctx, next_target);
    let selected = ctx.constant(at)?;
    let arrival_lower_boundary = ctx
        .structures
        .any(&c.mul(&lower, &c.eq(&state.arrival.lower, &selected)?)?)?;
    let arrival_upper_boundary = ctx
        .structures
        .any(&c.mul(&upper, &c.eq(&state.arrival.upper, &selected)?)?)?;
    let target_index = (command == Some(at)).then_some(next_target);
    let terminal = at == ctx.schedule.end_s;
    if target_index.is_none() && !arrival_lower_boundary && !arrival_upper_boundary && !terminal {
        return Ok(None);
    }
    Ok(Some(GateActuatorEvent {
        time_s: at,
        target_index,
        arrival_lower_boundary,
        arrival_upper_boundary,
        terminal,
    }))
}

/// Next strictly later target, arrival bound, or terminal event of a standalone walk.
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
    let (lower, upper) = bounds(ctx, state, time)?;
    let (_, fallback) = command(ctx, next_target);
    let a = &state.arrival;
    let time_s = earliest(
        ctx,
        fallback,
        &[(&lower, &a.lower), (&upper, &a.upper)],
        time,
    )?;
    event_at(ctx, state, (time, next_target), time_s)
}

/// Next strictly later target, arrival window, or terminal event of a coordinator.
///
/// A window strictly after `time` ends at its upper bound. A window that `time` lies inside
/// holds no further event.
pub(crate) fn next_coordinated_event<R: Runtime<DType = DType>>(
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
    let (ahead, _) = bounds(ctx, state, time)?;
    let (_, fallback) = command(ctx, next_target);
    let a = &state.arrival;
    let time_s = earliest(ctx, fallback, &[(&ahead, &a.upper)], time)?;
    event_at(ctx, state, (time, next_target), time_s)
}
