//! Certified closed-conduit barrel geometry inside checked interval calls.
//!
//! Every quantity is a solvr `Interval`. Callers open one `with_interval_arithmetic` call and
//! pass its context to these functions, so no function here adds a probe or a host read.

#[cfg(test)]
mod certificate_rows;
pub(crate) mod constants;
pub(crate) mod section;

use numr::dtype::DType;
use numr::ops::{
    CompareOps, CorrectlyRoundedOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use solvr::common::arithmetic::IntervalContext;

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
