//! Source physical residuals and independent persistent counter admission.

use crate::cpu::{
    CpuTensor,
    level_pool::{CpuPoolContext, CpuPoolSourceState, CpuPoolSourceStep},
};
use crate::storage::LevelPoolError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};

pub(super) fn check_state(
    context: &CpuPoolContext,
    state: &CpuPoolSourceState,
    steps: usize,
) -> Result<(), LevelPoolError> {
    for value in [&state.rain, &state.evaporation, &state.bottom_loss] {
        super::bounds::check_counter(context, value, steps)?;
    }
    context.nonnegative(
        &state.bottom_loss_demand,
        "cumulative potential bottom demand",
    )?;
    context.reject(
        &context
            .client
            .gt(&state.bottom_loss, &state.bottom_loss_demand)?,
        "cumulative actual bottom loss exceeds cumulative supplied potential demand",
    )?;
    if steps == 0 {
        context.reject(
            &context
                .client
                .ne(&state.bottom_loss_demand, &context.zero)?,
            "zero-history bottom demand must be zero",
        )?;
    }
    for (counter, bound) in [
        (&state.rain, &state.rain_bound),
        (&state.evaporation, &state.evaporation_bound),
        (&state.bottom_loss, &state.bottom_loss_bound),
        (&state.bottom_loss_demand, &state.bottom_loss_demand_bound),
    ] {
        context.nonnegative(bound, "cumulative source addition-error bound")?;
        context.reject(
            &context.client.mul(
                &context.client.eq(counter, &context.zero)?,
                &context.client.ne(bound, &context.zero)?,
            )?,
            "zero source throughput requires zero addition-error bound",
        )?;
    }
    Ok(())
}

pub(super) struct Interval {
    pub residual: CpuTensor,
    pub consistency: CpuTensor,
    pub bound: CpuTensor,
    pub scale: CpuTensor,
}

pub(super) fn interval(
    context: &CpuPoolContext,
    source: &CpuPoolSourceStep,
    a: &CpuTensor,
    b: &CpuTensor,
    legacy: &Interval,
    out_witness: &CpuTensor,
    in_witness: &CpuTensor,
) -> Result<Interval, LevelPoolError> {
    let c = &context.client;
    let d = c.sub(&legacy.residual, &source.rain_volume_m3)?;
    let e = c.add(&d, &source.bottom_loss_cap.actual_volume_m3)?;
    let residual = c.add(&e, &source.evaporation_cap.actual_volume_m3)?;
    let sum1 = c.add(&source.rain_witness_m3, &source.bottom_loss_witness_m3)?;
    let sum2 = c.add(&sum1, &source.evaporation_witness_m3)?;
    let sum3 = c.add(&sum2, out_witness)?;
    let sum4 = c.add(&sum3, in_witness)?;
    let consistency = c.add(&residual, &sum4)?;
    let mut bound = super::bounds::rounding(context, a)?;
    for value in [
        b,
        &legacy.residual,
        &d,
        &e,
        &residual,
        &sum1,
        &sum2,
        &sum3,
        &sum4,
        &consistency,
    ] {
        bound = super::bounds::add(context, &bound, &super::bounds::rounding(context, value)?)?;
    }
    let scale = c.maximum(
        &legacy.scale,
        &c.maximum(
            &source.rain_volume_m3,
            &c.maximum(
                &source.bottom_loss_cap.actual_volume_m3,
                &source.evaporation_cap.actual_volume_m3,
            )?,
        )?,
    )?;
    Ok(Interval {
        residual: c.where_cond(&source.active, &residual, &legacy.residual)?,
        consistency: c.where_cond(&source.active, &consistency, &legacy.consistency)?,
        bound: c.where_cond(&source.active, &bound, &legacy.bound)?,
        scale: c.where_cond(&source.active, &scale, &legacy.scale)?,
    })
}
