//! Interval operation from the start and end points of one event-free interval.

use super::{TensorOperatingSourceContext, TensorOperatingSourcePoint};
use crate::structures::StructureError;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps, ScalarOps, TwoSumOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Interval operation tensors in complete original structure layout.
#[derive(Clone, Debug)]
pub(crate) struct TensorOperatingSourceInterval<R: Runtime<DType = DType>> {
    /// Gate opening the law takes: the rounded midpoint, or canonical zero when closed at both ends.
    pub point: TensorOperatingSourcePoint<R>,
    /// Exact distance from the rounded midpoint to the mean of the represented endpoint openings.
    pub rounding: Tensor<R>,
}

/// Midpoint gate openings, closure at both ends, larger endpoint error, and held start pump operation.
pub(crate) fn combine<R: Runtime<DType = DType>>(
    context: &TensorOperatingSourceContext<R>,
    start: &TensorOperatingSourcePoint<R>,
    end: &TensorOperatingSourcePoint<R>,
) -> Result<TensorOperatingSourceInterval<R>, StructureError>
where
    R::Client: BinaryOps<R> + ConditionalOps<R> + ScalarOps<R> + TwoSumOps<R> + UnaryOps<R>,
{
    let structures = &context.structures;
    let c = &structures.client;
    let (sum, residual) = c.two_sum(&start.opening, &end.opening)?;
    let closed = c.mul(&start.closed, &end.closed)?;
    let midpoint = c.mul_scalar(&sum, 0.5)?;
    let rounding = c.abs(&c.mul_scalar(&residual, 0.5)?)?;
    let mut point = start.clone();
    point.opening = c.where_cond(&closed, &structures.zero, &midpoint)?;
    point.absolute_error = c.where_cond(
        &closed,
        &structures.zero,
        &c.maximum(&start.absolute_error, &end.absolute_error)?,
    )?;
    point.closed = closed.clone();
    Ok(TensorOperatingSourceInterval {
        point,
        rounding: c.where_cond(&closed, &structures.zero, &rounding)?,
    })
}
