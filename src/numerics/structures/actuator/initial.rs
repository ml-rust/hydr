//! Canonical original assumed actuator positions and first motion episodes.

use super::{arrival::arrival, evaluate::phase};
use crate::cpu::structures::actuator::{
    CpuActuatorAnchor, CpuActuatorContext, CpuActuatorPosition, CpuActuatorState,
    record::TensorRecord,
};
use crate::structures::StructureError;
use numr::ops::{CompareOps, ConditionalOps};

pub(crate) fn initial(ctx: &CpuActuatorContext) -> Result<CpuActuatorState, StructureError> {
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
    let position = CpuActuatorPosition {
        center: center.clone(),
        lower: center.clone(),
        upper: center,
        error: ctx.zero.clone(),
    };
    let anchor = CpuActuatorAnchor {
        epoch: ctx.constant(ctx.schedule.initial_time_s)?,
        position: position.clone(),
    };
    let arrival = arrival(ctx, &anchor, &target)?;
    let active = c.ne(&position.center, &target)?;
    let mut state = CpuActuatorState::zero(ctx);
    state.phase = phase(ctx, &arrival, ctx.schedule.initial_time_s)?;
    state.anchor = anchor;
    state.target = target;
    state.position = position;
    state.arrival = arrival;
    state.episodes = active.clone();
    state.active_valid = active;
    Ok(state)
}
