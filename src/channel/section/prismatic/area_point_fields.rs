use super::{
    area_point_input::Inputs,
    area_point_inverse,
    area_point_query::PrismaticAreaPointBandCoordinates,
    area_query::fields::{self as area_query_fields, LocalFields},
    band_geometry::{BandRows, hull, masked},
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
    let LocalFields {
        forward_area: forward,
        top_width,
        integral,
    } = area_query_fields::polynomials(
        a,
        context,
        rows,
        candidate,
        &coordinates.depth_above_start,
    )?;
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
