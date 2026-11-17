use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

/// Scalar I64 code on the client's device.
pub(super) fn code<R>(a: &Arithmetic<'_, R>, value: i64) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(Tensor::full_scalar(
        &[],
        DType::I64,
        value as f64,
        a.client().device(),
    )?)
}

/// Numeric zero-one mask of the I64 entries equal to `value`.
pub(super) fn equals<R>(
    a: &Arithmetic<'_, R>,
    values: &Tensor<R>,
    value: i64,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    flag(a, &a.client().eq(values, &code(a, value)?)?)
}

/// Converts a comparison result to a numeric zero-one mask.
pub(super) fn flag<R>(a: &Arithmetic<'_, R>, condition: &Tensor<R>) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    Ok(c.where_cond(
        condition,
        &Tensor::ones(condition.shape(), a.dtype(), c.device())?,
        &Tensor::zeros(condition.shape(), a.dtype(), c.device())?,
    )?)
}

/// Zero-one mask of the zero entries of a zero-one mask.
pub(super) fn not<R>(a: &Arithmetic<'_, R>, mask: &Tensor<R>) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    flag(a, &c.eq(mask, &Tensor::zeros(&[], a.dtype(), c.device())?)?)
}

pub(super) fn copy<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(a.bounds(value.lower(), value.upper())?)
}

pub(super) fn assign<R>(
    a: &mut Arithmetic<'_, R>,
    target: &Interval<R>,
    positions: &Tensor<R>,
    value: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let lower = c.index_put(target.lower(), 0, positions, value.lower())?;
    let upper = c.index_put(target.upper(), 0, positions, value.upper())?;
    Ok(a.bounds(&lower, &upper)?)
}
