use super::super::{
    band_geometry::{BandRows, masked},
    compilation_bounds::nonnegative,
    context::Context,
    query_input::{Arithmetic, QueryResult},
};
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(in crate::channel::section::prismatic) struct LocalFields<R: Runtime<DType = DType>> {
    pub forward_area: Interval<R>,
    pub top_width: Interval<R>,
    pub integral: Interval<R>,
}

/// Callers prove each active local depth lies inside its physical band.
/// Inactive local operands must remain safe for eager arithmetic.
pub(in crate::channel::section::prismatic) fn polynomials<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    rows: &BandRows<R>,
    candidate: &Tensor<R>,
    local: &Interval<R>,
) -> QueryResult<LocalFields<R>>
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
    let zero = &context.source.zero;
    let width = masked(a, candidate, &rows.start_width, zero)?;
    let slope = masked(a, candidate, &rows.slope, zero)?;
    let area = masked(a, candidate, &rows.start_area, zero)?;
    let integral = masked(a, candidate, &rows.start_integral, zero)?;
    let square = a.mul(local, local)?;
    let slope_depth = a.mul(&slope, local)?;
    let top_width = a.add(&width, &slope_depth)?;
    let top_width = nonnegative(a, top_width, zero)?;
    let two = c.mul_scalar(&context.source.one, 2.0)?;
    let six = c.mul_scalar(&context.source.one, 6.0)?;
    let width_depth = a.mul(&width, local)?;
    // Multiply slope before depth again, preserving pointed subnormal area scales.
    let slope_square = a.mul(&slope_depth, local)?;
    let half_slope = a.div_positive(&slope_square, &two)?;
    let area_width = a.add(&area, &width_depth)?;
    let forward = a.add(&area_width, &half_slope)?;
    let forward = nonnegative(a, forward, zero)?;
    let area_depth = a.mul(&area, local)?;
    let width_square = a.mul(&width, &square)?;
    let half_width = a.div_positive(&width_square, &two)?;
    let slope_cube = a.mul(&slope_square, local)?;
    let sixth_slope = a.div_positive(&slope_cube, &six)?;
    let integral_area = a.add(&integral, &area_depth)?;
    let integral_width = a.add(&integral_area, &half_width)?;
    let integral = a.add(&integral_width, &sixth_slope)?;
    let integral = nonnegative(a, integral, zero)?;
    Ok(LocalFields {
        forward_area: forward,
        top_width,
        integral,
    })
}
