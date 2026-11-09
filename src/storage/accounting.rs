//! Aggregate reductions independent of per-element physical admission.

use super::{LevelPoolError, LevelPoolTotals};
use crate::cpu::{
    CpuTensor,
    level_pool::{CpuPoolContext, CpuPoolState, CpuPoolStep},
};

fn outward_add(a: f64, b: f64) -> Result<f64, LevelPoolError> {
    if a == 0.0 {
        return Ok(b);
    }
    if b == 0.0 {
        return Ok(a);
    }
    crate::soil::admission::upward_add(a, b).map_err(|e| LevelPoolError::numerical(e.to_string()))
}

fn reduction_bound(context: &CpuPoolContext, sums: &[f64]) -> Result<f64, LevelPoolError> {
    let n = context.elements.len();
    if n <= 1 {
        return Ok(0.0);
    }
    let additions = n
        .checked_add(7)
        .ok_or_else(|| LevelPoolError::input("aggregate reduction count overflows"))?;
    if additions as u128 > (1u128 << 53) {
        return Err(LevelPoolError::input(
            "aggregate reduction count exceeds exact F64 integer range",
        ));
    }
    let x = additions as f64 * f64::EPSILON;
    let b = |v: f64| {
        crate::soil::admission::rounding_bound(v)
            .map_err(|e| LevelPoolError::numerical(e.to_string()))
    };
    let d0 = 1.0 - x;
    let denominator = d0 - b(d0)?;
    if !(0.0..1.0).contains(&x) || denominator <= 0.0 {
        return Err(LevelPoolError::input(
            "aggregate reduction uncertainty denominator is not positive",
        ));
    }
    let fraction = x / denominator;
    let factor = fraction + b(fraction)?;
    let mut total = 0.0;
    for value in sums {
        if *value == 0.0 {
            continue;
        }
        let bound = crate::soil::admission::upward_mul(*value, factor)
            .map_err(|e| LevelPoolError::numerical(e.to_string()))?;
        total = outward_add(total, bound)?;
    }
    Ok(total)
}

fn sum_bound(
    context: &CpuPoolContext,
    value: &CpuTensor,
    quantity: &str,
) -> Result<f64, LevelPoolError> {
    let sum = context.sum(value, quantity)?;
    outward_add(sum, reduction_bound(context, &[sum])?)
}

pub(crate) fn check_scalar(
    context: &CpuPoolContext,
    residual: f64,
    scale: f64,
    quantity: &str,
) -> Result<(), LevelPoolError> {
    let tolerance = context.config.volume_absolute + context.config.volume_relative * scale;
    if !residual.is_finite()
        || !scale.is_finite()
        || !tolerance.is_finite()
        || residual.abs() > tolerance
    {
        return Err(LevelPoolError::numerical(format!(
            "{quantity},residual={residual},scale={scale},tolerance={tolerance}: rescale pool quantities or select an admitted interval"
        )));
    }
    Ok(())
}

pub(crate) fn totals(
    context: &CpuPoolContext,
    state: &CpuPoolState,
) -> Result<LevelPoolTotals, LevelPoolError> {
    let initial = context.sum(&state.initial_volume_m3, "aggregate original volume")?;
    let volume = context.sum(&state.volume_m3, "aggregate current volume")?;
    let incoming = context.sum(
        &state.cumulative_incoming_volume_m3,
        "aggregate cumulative incoming",
    )?;
    let outgoing = context.sum(
        &state.cumulative_outgoing_volume_m3,
        "aggregate cumulative outgoing",
    )?;
    let residual = (volume - initial) - incoming + outgoing;
    let scale = initial.max(volume).max(incoming).max(outgoing);
    check_scalar(
        context,
        residual,
        scale,
        "aggregate cumulative physical residual",
    )?;
    Ok(LevelPoolTotals {
        initial_volume_m3: initial,
        volume_m3: volume,
        cumulative_incoming_volume_m3: incoming,
        cumulative_outgoing_volume_m3: outgoing,
        incoming_roundoff_bound_m3: sum_bound(
            context,
            &state.incoming_roundoff_bound_m3,
            "aggregate incoming addition bound",
        )?,
        outgoing_roundoff_bound_m3: sum_bound(
            context,
            &state.outgoing_roundoff_bound_m3,
            "aggregate outgoing addition bound",
        )?,
        residual_m3: residual,
        physical_scale_m3: scale,
        reduction_bound_m3: reduction_bound(context, &[initial, volume, incoming, outgoing])?,
    })
}

pub(crate) struct StepTotals {
    pub before: f64,
    pub after: f64,
    pub incoming: f64,
    pub outgoing: f64,
    pub residual: f64,
    pub scale: f64,
    pub bound: f64,
}
pub(crate) fn step_totals(
    context: &CpuPoolContext,
    step: &CpuPoolStep,
) -> Result<StepTotals, LevelPoolError> {
    let before = context.sum(&step.old_volume_m3, "aggregate interval original volume")?;
    let after = context.sum(&step.volume_m3, "aggregate interval accepted volume")?;
    let incoming = context.sum(&step.incoming_volume_m3, "aggregate interval incoming")?;
    let outgoing = context.sum(&step.outgoing_volume_m3, "aggregate interval outgoing")?;
    let residual = (after - before) - incoming + outgoing;
    let scale = before.max(after).max(incoming).max(outgoing);
    check_scalar(
        context,
        residual,
        scale,
        "aggregate interval physical residual",
    )?;
    Ok(StepTotals {
        before,
        after,
        incoming,
        outgoing,
        residual,
        scale,
        bound: reduction_bound(context, &[before, after, incoming, outgoing])?,
    })
}

pub(crate) fn read(value: &CpuTensor, count: usize) -> Result<Vec<f64>, LevelPoolError> {
    if value.shape() != [count] {
        return Err(LevelPoolError::numerical(format!(
            "projection shape={:?}: require {count} ordered pool values",
            value.shape()
        )));
    }
    Ok(value.try_to_vec::<f64>()?)
}
