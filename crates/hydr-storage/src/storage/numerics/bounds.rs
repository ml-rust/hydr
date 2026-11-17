//! Outward arithmetic diagnostics and capacity-derived counter ceilings.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::storage::LevelPoolError;
use crate::storage::tensor::TensorPoolContext;

pub(crate) fn rounding<R>(
    context: &TensorPoolContext<R>,
    value: &Tensor<R>,
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
    let bound = c.add_scalar(
        &c.mul_scalar(&c.abs(value)?, f64::EPSILON)?,
        f64::from_bits(1),
    )?;
    context.finite(&bound, "arithmetic rounding bound")?;
    Ok(bound)
}
pub(crate) fn add<R>(
    context: &TensorPoolContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
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
    let sum = c.add(a, b)?;
    context.finite(&sum, "absolute bound sum")?;
    let result = c.add(&sum, &rounding(context, &sum)?)?;
    context.finite(&result, "outward absolute bound")?;
    Ok(result)
}
pub(crate) fn accumulate<R>(
    context: &TensorPoolContext<R>,
    previous: &Tensor<R>,
    witness: &Tensor<R>,
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
    let inactive = c.eq(witness, &context.zero)?;
    let magnitude = c.abs(witness)?;
    let active_previous = c.where_cond(&inactive, &context.zero, previous)?;
    let padded = add(context, &active_previous, &magnitude)?;
    Ok(c.where_cond(&inactive, previous, &padded)?)
}

/// Theoretical overflow selects an unrestricted finite-counter check, never a physical infinity.
pub(crate) fn check_counter<R>(
    context: &TensorPoolContext<R>,
    counter: &Tensor<R>,
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
    context.nonnegative(counter, "cumulative transfer counter")?;
    let c = &context.client;
    if steps == 0 {
        return context.reject(
            &c.ne(counter, &context.zero)?,
            "zero-history cumulative counter",
        );
    }
    if steps == 1 {
        return context.reject(
            &c.gt(counter, &context.maximum)?,
            "single-step counter exceeds capacity",
        );
    }
    if steps as u128 > (1u128 << 53) {
        return Ok(());
    }
    let x = steps as f64 * f64::EPSILON;
    let d0 = 1.0 - x;
    if !(0.0..1.0).contains(&x) {
        return Ok(());
    }
    let denominator = d0 - (f64::EPSILON * d0.abs() + f64::from_bits(1));
    if !denominator.is_finite() || denominator <= 0.0 {
        return Ok(());
    }
    // Positive upper-bound overflow is tracked separately from actual state validity.
    let p = c.mul_scalar(&context.maximum, steps as f64)?;
    let p_bound = c.add_scalar(&c.mul_scalar(&c.abs(&p)?, f64::EPSILON)?, f64::from_bits(1))?;
    let upper = c.add(&p, &p_bound)?;
    let q = c.div_scalar(&upper, denominator)?;
    let q_bound = c.add_scalar(&c.mul_scalar(&c.abs(&q)?, f64::EPSILON)?, f64::from_bits(1))?;
    let ceiling = c.add(&q, &q_bound)?;
    let unbounded = c.isinf(&ceiling)?;
    let ceiling = c.where_cond(
        &unbounded,
        &c.add_scalar(&context.zero, f64::MAX)?,
        &ceiling,
    )?;
    context.finite(&ceiling, "finite theoretical counter ceiling")?;
    context.reject(
        &c.gt(counter, &ceiling)?,
        "claimed cumulative transfer exceeds capacity-and-step ceiling",
    )
}
