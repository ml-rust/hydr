//! Aggregate reductions independent of per-element physical admission.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

use super::{LevelPoolError, LevelPoolTotals};
use crate::storage::tensor::{TensorPoolContext, TensorPoolState, TensorPoolStep};

pub(super) fn outward_add(a: f64, b: f64) -> Result<f64, LevelPoolError> {
    if a == 0.0 {
        return Ok(b);
    }
    if b == 0.0 {
        return Ok(a);
    }
    crate::soil::admission::upward_add(a, b).map_err(|e| LevelPoolError::numerical(e.to_string()))
}

pub(super) fn reduction_bound<R>(
    context: &TensorPoolContext<R>,
    sums: &[f64],
) -> Result<f64, LevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
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

pub(crate) fn sum_bound<R>(
    context: &TensorPoolContext<R>,
    value: &Tensor<R>,
    quantity: &str,
) -> Result<f64, LevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    let sum = context.sum(value, quantity)?;
    outward_add(sum, reduction_bound(context, &[sum])?)
}

pub(crate) fn check_scalar<R>(
    context: &TensorPoolContext<R>,
    residual: f64,
    scale: f64,
    quantity: &str,
) -> Result<(), LevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
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

pub(crate) fn totals<R>(
    context: &TensorPoolContext<R>,
    state: &TensorPoolState<R>,
) -> Result<LevelPoolTotals, LevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
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
    let sources = super::source_projection::totals(context, state.sources.as_ref())?;
    let mut residual = (volume - initial) - incoming + outgoing;
    let mut scale = initial.max(volume).max(incoming).max(outgoing);
    let mut sums = vec![initial, volume, incoming, outgoing];
    if let Some(s) = &sources {
        residual = ((residual - s.cumulative_rain_volume_m3) + s.cumulative_bottom_loss_volume_m3)
            + s.cumulative_evaporation_volume_m3;
        scale = scale
            .max(s.cumulative_rain_volume_m3)
            .max(s.cumulative_bottom_loss_volume_m3)
            .max(s.cumulative_evaporation_volume_m3);
        sums.extend([
            s.cumulative_rain_volume_m3,
            s.cumulative_bottom_loss_volume_m3,
            s.cumulative_evaporation_volume_m3,
        ]);
    }
    check_scalar(
        context,
        residual,
        scale,
        "aggregate cumulative physical residual",
    )?;
    Ok(LevelPoolTotals {
        sources,
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
        reduction_bound_m3: reduction_bound(context, &sums)?,
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
pub(crate) fn step_totals<R>(
    context: &TensorPoolContext<R>,
    step: &TensorPoolStep<R>,
) -> Result<StepTotals, LevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    let before = context.sum(&step.old_volume_m3, "aggregate interval original volume")?;
    let after = context.sum(&step.volume_m3, "aggregate interval accepted volume")?;
    let incoming = context.sum(&step.incoming_volume_m3, "aggregate interval incoming")?;
    let outgoing = context.sum(&step.outgoing_volume_m3, "aggregate interval outgoing")?;
    let source = super::source_projection::step_totals(context, step.source.as_ref())?;
    let mut residual = (after - before) - incoming + outgoing;
    let mut scale = before.max(after).max(incoming).max(outgoing);
    let mut sums = vec![before, after, incoming, outgoing];
    if step.source.is_some() {
        residual = ((residual - source.rain_volume_m3) + source.bottom_loss_volume_m3)
            + source.evaporation_volume_m3;
        scale = scale
            .max(source.rain_volume_m3)
            .max(source.bottom_loss_volume_m3)
            .max(source.evaporation_volume_m3);
        sums.extend([
            source.rain_volume_m3,
            source.bottom_loss_volume_m3,
            source.evaporation_volume_m3,
        ]);
    }
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
        bound: reduction_bound(context, &sums)?,
    })
}

pub(crate) fn read<R>(value: &Tensor<R>, count: usize) -> Result<Vec<f64>, LevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    if value.shape() != [count] {
        return Err(LevelPoolError::numerical(format!(
            "projection shape={:?}: require {count} ordered pool values",
            value.shape()
        )));
    }
    Ok(value.try_to_vec::<f64>()?)
}
