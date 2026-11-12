use super::{
    area_point_input::Inputs,
    area_point_inverse,
    area_point_query::PrismaticAreaPointBandCoordinates,
    band_geometry::{BandRows, hull, masked},
    compilation_bounds::nonnegative,
    context::Context,
    query::Fields,
    query_input::{Arithmetic, QueryResult, reject},
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};

pub(super) fn geometry<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    rows: &BandRows<R>,
    inputs: &Inputs<R>,
    coordinates: &PrismaticAreaPointBandCoordinates<R>,
) -> QueryResult<Fields<R>>
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
    let candidate = &coordinates.candidate;
    let local = &coordinates.depth_above_start;
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
    let absolute = area_point_inverse::absolute_depth(a, context, rows, coordinates)?;
    let depth = hull(
        a,
        candidate,
        &absolute,
        &inputs.wet,
        rows.cap_depth.upper(),
        zero,
    )?;
    let forward = hull(
        a,
        candidate,
        &forward,
        &inputs.wet,
        rows.cap_area.upper(),
        zero,
    )?;
    let top_width = hull(
        a,
        candidate,
        &top_width,
        &inputs.wet,
        rows.cap_width.upper(),
        zero,
    )?;
    let integral = hull(
        a,
        candidate,
        &integral,
        &inputs.wet,
        rows.cap_integral.upper(),
        zero,
    )?;
    reject(
        context,
        &inputs.owners,
        &c.mul(&inputs.wet, &c.le(top_width.lower(), zero)?)?,
        true,
        "wet width has no positive lower bound. Check profile conditioning",
    )?;
    let denominator = masked(a, &inputs.wet, &top_width, &context.source.one)?;
    let numerator = a.expansion_interval(&inputs.areas)?;
    let hydraulic_depth = a.div_positive_interval(&numerator, &denominator)?;
    let hydraulic_depth = a.bounds(
        &c.maximum(hydraulic_depth.lower(), zero)?,
        &c.minimum(hydraulic_depth.upper(), depth.upper())?,
    )?;
    let hydraulic_depth = masked(a, &inputs.wet, &hydraulic_depth, zero)?;
    Ok(Fields {
        depth,
        area: forward,
        top_width,
        integral,
        hydraulic_depth,
    })
}
