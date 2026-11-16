//! Expansion types and helpers shared by the two-part conserved updates.

use crate::channel::flow::{
    ChannelError, conserved_point::ConservedPoint, surveyed_transport::error::arithmetic_error,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{
    ArithmeticResult, CertifiedExpansion, IntervalContext, with_interval_arithmetic,
};

/// Client operations that certified expansion arithmetic requires.
pub(crate) trait ExpansionClient<R: Runtime<DType = DType>>:
    TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + ScalarOps<R> + TwoSumOps<R>
{
}

impl<R, C> ExpansionClient<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + ScalarOps<R> + TwoSumOps<R>,
{
}

pub(in crate::channel::flow) type Arithmetic<'a, R> =
    IntervalContext<'a, R, <R as Runtime>::Client>;

/// Expansion center `(high, low)` and its absolute error radius.
#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct Parts<R: Runtime<DType = DType>> {
    pub high: Tensor<R>,
    pub low: Tensor<R>,
    pub bound: Tensor<R>,
}

/// Run one checked expansion block and map its arithmetic error.
pub(in crate::channel::flow) fn expanded<R, T>(
    client: &R::Client,
    stage: &'static str,
    build: impl for<'a> FnOnce(&mut Arithmetic<'a, R>) -> ArithmeticResult<T>,
) -> Result<T, ChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    with_interval_arithmetic::<R, _, _, _>(client, DType::F64, stage, build)
        .map_err(arithmetic_error)
}

/// Exact expansion of a normalized `(high, low)` pair with zero radius.
pub(in crate::channel::flow) fn exact<R>(
    a: &mut Arithmetic<'_, R>,
    high: &Tensor<R>,
    low: &Tensor<R>,
) -> ArithmeticResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let zero = Tensor::zeros(high.shape(), DType::F64, a.client().device())?;
    exact_over(a, high, low, &zero)
}

/// Exact expansion that reuses a caller-owned zero radius of the same shape.
pub(in crate::channel::flow) fn exact_over<R>(
    a: &mut Arithmetic<'_, R>,
    high: &Tensor<R>,
    low: &Tensor<R>,
    zero: &Tensor<R>,
) -> ArithmeticResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    a.expansion_bounds(high, low, zero)
}

pub(in crate::channel::flow) fn bounded<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Parts<R>,
) -> ArithmeticResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    a.expansion_bounds(&value.high, &value.low, &value.bound)
}

/// Selected center of an expansion. Its radius is recorded apart.
pub(in crate::channel::flow) fn center<R: Runtime<DType = DType>>(
    value: &CertifiedExpansion<R>,
) -> ConservedPoint<R> {
    ConservedPoint {
        high: value.high().clone(),
        low: value.low().clone(),
    }
}

/// Center parts and radius of an expansion.
pub(in crate::channel::flow) fn parts<R: Runtime<DType = DType>>(
    value: &CertifiedExpansion<R>,
) -> Parts<R> {
    Parts {
        high: value.high().clone(),
        low: value.low().clone(),
        bound: value.abs_error().clone(),
    }
}

/// Upper bound on the magnitude of every value inside the expansion.
pub(in crate::channel::flow) fn magnitude<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CertifiedExpansion<R>,
) -> ArithmeticResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let interval = a.expansion_interval(value)?;
    let c = a.client();
    Ok(c.maximum(&c.abs(interval.lower())?, &c.abs(interval.upper())?)?)
}

/// Difference of the left and right face points of every cell.
pub(in crate::channel::flow) fn transfer<R>(
    a: &mut Arithmetic<'_, R>,
    face: &CertifiedExpansion<R>,
    count: usize,
) -> ArithmeticResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let zero = Tensor::zeros(&[count], DType::F64, a.client().device())?;
    let incoming = exact_over(
        a,
        &face.high().narrow(0, 0, count)?,
        &face.low().narrow(0, 0, count)?,
        &zero,
    )?;
    let outgoing = exact_over(
        a,
        &face.high().narrow(0, 1, count)?,
        &face.low().narrow(0, 1, count)?,
        &zero,
    )?;
    a.expansion_sub(&incoming, &outgoing)
}
