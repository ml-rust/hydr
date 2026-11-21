//! Certified closed-conduit barrel geometry inside checked interval calls.
//!
//! Every quantity is a solvr `Interval`. Callers open one `with_interval_arithmetic` call and
//! pass its context to these functions, so no function here adds a probe or a host read.

#[cfg(test)]
mod certificate_rows;
pub(crate) mod constants;
pub(crate) mod critical;
pub(crate) mod normal;
mod roots;
pub(crate) mod section;

pub(crate) use roots::BarrelDepths;

use numr::dtype::DType;
use numr::ops::{
    CompareOps, ConditionalOps, CorrectlyRoundedOps, NextRepresentableOps, ScalarOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval, IntervalContext};

/// Interval context on the barrel batch's client.
pub(crate) type Arithmetic<'a, R> = IntervalContext<'a, R, <R as Runtime>::Client>;

/// Client operations that barrel geometry needs.
pub(crate) trait BarrelClient<R: Runtime<DType = DType>>:
    TensorOps<R>
    + CompareOps<R>
    + NextRepresentableOps<R>
    + ScalarOps<R>
    + TwoSumOps<R>
    + CorrectlyRoundedOps<R>
    + RuntimeClient<R>
{
}

impl<R, C> BarrelClient<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + CorrectlyRoundedOps<R>
        + RuntimeClient<R>,
{
}

/// Takes `on` where `mask` is 1 and `off` where it is 0.
pub(crate) fn choose<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    on: &Interval<R>,
    off: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let lower = c.where_cond(mask, on.lower(), off.lower())?;
    let upper = c.where_cond(mask, on.upper(), off.upper())?;
    a.bounds(&lower, &upper)
}
