//! Canonical original assumed actuator positions and first motion episodes.

use super::{arrival::arrival, evaluate::phase};
use crate::structures::StructureError;
use crate::tensor::structures::actuator::{
    TensorActuatorAnchor, TensorActuatorContext, TensorActuatorPosition, TensorActuatorState,
    record::TensorRecord,
};
use numr::dtype::DType;
use numr::ops::{CompareOps, ConditionalOps};
use numr::runtime::Runtime;

pub(crate) fn initial<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
) -> Result<TensorActuatorState<R>, StructureError>
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
    let target = ctx.tensor(
        &ctx.initial_states
            .iter()
            .map(|s| s.target_opening_m)
            .collect::<Vec<_>>(),
    )?;
    let original = ctx.tensor(
        &ctx.initial_states
            .iter()
            .map(|s| s.opening_m)
            .collect::<Vec<_>>(),
    )?;
    let center = c.where_cond(&c.eq(&original, &target)?, &target, &original)?;
    let position = TensorActuatorPosition {
        center: center.clone(),
        lower: center.clone(),
        upper: center,
        error: ctx.zero.clone(),
    };
    let anchor = TensorActuatorAnchor {
        epoch: ctx.constant(ctx.schedule.initial_time_s)?,
        position: position.clone(),
    };
    let arrival = arrival(ctx, &anchor, &target)?;
    let active = c.ne(&position.center, &target)?;
    let mut state = TensorActuatorState::zero(ctx);
    state.phase = phase(ctx, &arrival, ctx.schedule.initial_time_s)?;
    state.anchor = anchor;
    state.target = target;
    state.position = position;
    state.arrival = arrival;
    state.episodes = active.clone();
    state.active_valid = active;
    Ok(state)
}
