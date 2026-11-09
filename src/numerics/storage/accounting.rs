//! Independent per-element physical drift gates, separate from witnesses.

use crate::cpu::{
    CpuTensor,
    level_pool::{CpuPoolContext, CpuPoolState},
};
use crate::storage::LevelPoolError;
use numr::ops::{BinaryOps, CompareOps, ScalarOps, UnaryOps};

pub(crate) fn check_residual(
    context: &CpuPoolContext,
    residual: &CpuTensor,
    scale: &CpuTensor,
    quantity: &str,
) -> Result<(), LevelPoolError> {
    context.finite(residual, quantity)?;
    context.nonnegative(scale, "physical residual scale")?;
    let c = &context.client;
    let tolerance = c.add_scalar(
        &c.mul_scalar(scale, context.config.volume_relative)?,
        context.config.volume_absolute,
    )?;
    context.finite(&tolerance, "dimensional physical tolerance")?;
    context.reject(&c.gt(&c.abs(residual)?, &tolerance)?, quantity)
}
pub(crate) fn residual(
    context: &CpuPoolContext,
    state: &CpuPoolState,
) -> Result<CpuTensor, LevelPoolError> {
    let c = &context.client;
    Ok(c.add(
        &c.sub(
            &c.sub(&state.volume_m3, &state.initial_volume_m3)?,
            &state.cumulative_incoming_volume_m3,
        )?,
        &state.cumulative_outgoing_volume_m3,
    )?)
}
pub(crate) fn check_state(
    context: &CpuPoolContext,
    state: &CpuPoolState,
    steps: usize,
) -> Result<(), LevelPoolError> {
    context.range(&state.volume_m3, "current pool volume")?;
    context.range(&state.initial_volume_m3, "initial pool volume")?;
    super::bounds::check_counter(context, &state.cumulative_incoming_volume_m3, steps)?;
    super::bounds::check_counter(context, &state.cumulative_outgoing_volume_m3, steps)?;
    context.nonnegative(
        &state.incoming_roundoff_bound_m3,
        "incoming addition-error bound",
    )?;
    context.nonnegative(
        &state.outgoing_roundoff_bound_m3,
        "outgoing addition-error bound",
    )?;
    let c = &context.client;
    for (counter, bound) in [
        (
            &state.cumulative_incoming_volume_m3,
            &state.incoming_roundoff_bound_m3,
        ),
        (
            &state.cumulative_outgoing_volume_m3,
            &state.outgoing_roundoff_bound_m3,
        ),
    ] {
        context.reject(
            &c.mul(&c.eq(counter, &context.zero)?, &c.ne(bound, &context.zero)?)?,
            "zero-throughput history requires zero addition-error bound",
        )?;
    }
    let scale = c.maximum(
        &c.maximum(&state.volume_m3, &state.initial_volume_m3)?,
        &c.maximum(
            &state.cumulative_incoming_volume_m3,
            &state.cumulative_outgoing_volume_m3,
        )?,
    )?;
    check_residual(
        context,
        &residual(context, state)?,
        &scale,
        "per-element cumulative physical water residual",
    )
}
