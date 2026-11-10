//! Independent per-element physical drift gates, separate from witnesses.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::storage::LevelPoolError;
use crate::tensor::level_pool::{TensorPoolContext, TensorPoolState};

pub(crate) fn check_residual<R>(
    context: &TensorPoolContext<R>,
    residual: &Tensor<R>,
    scale: &Tensor<R>,
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
        + SortingOps<R>,
{
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
pub(crate) fn residual<R>(
    context: &TensorPoolContext<R>,
    state: &TensorPoolState<R>,
) -> Result<Tensor<R>, LevelPoolError>
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
        + SortingOps<R>,
{
    let c = &context.client;
    let residual = c.add(
        &c.sub(
            &c.sub(&state.volume_m3, &state.initial_volume_m3)?,
            &state.cumulative_incoming_volume_m3,
        )?,
        &state.cumulative_outgoing_volume_m3,
    )?;
    if let Some(s) = &state.sources {
        return Ok(c.add(
            &c.add(&c.sub(&residual, &s.rain)?, &s.bottom_loss)?,
            &s.evaporation,
        )?);
    }
    Ok(residual)
}
pub(crate) fn check_state<R>(
    context: &TensorPoolContext<R>,
    state: &TensorPoolState<R>,
    steps: usize,
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
        + SortingOps<R>,
{
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
    let mut scale = c.maximum(
        &c.maximum(&state.volume_m3, &state.initial_volume_m3)?,
        &c.maximum(
            &state.cumulative_incoming_volume_m3,
            &state.cumulative_outgoing_volume_m3,
        )?,
    )?;
    match (&context.source_policies, &state.sources) {
        (Some(_), Some(s)) => {
            super::source_accounting::check_state(context, s, steps)?;
            scale = c.maximum(
                &scale,
                &c.maximum(&s.rain, &c.maximum(&s.bottom_loss, &s.evaporation)?)?,
            )?;
        }
        (None, None) => {}
        _ => {
            return Err(LevelPoolError::input(
                "source policy presence differs from persistent source accounting",
            ));
        }
    }
    check_residual(
        context,
        &residual(context, state)?,
        &scale,
        "per-element cumulative physical water residual",
    )
}
