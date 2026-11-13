use super::{
    area_point_input::{Inputs, difference},
    area_point_query::PrismaticAreaPointBandCoordinates,
    area_query::inverse::{self as area_query_inverse, LocalInverse},
    band_geometry::{BandRows, masked},
    context::Context,
    query_input::{Arithmetic, QueryResult, reject},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ReduceOps, ScalarOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use solvr::common::arithmetic::Interval;

#[cfg(test)]
use super::area_query::inverse::pointed_depth;

pub(super) fn recover<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    rows: &BandRows<R>,
    inputs: &Inputs<R>,
) -> QueryResult<PrismaticAreaPointBandCoordinates<R>>
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
    let count = inputs.owners.numel();
    let valid = c.mul(&rows.valid, &inputs.wet.reshape(&[count, 1])?)?;
    let high = c.where_cond(&valid, &inputs.areas.high().reshape(&[count, 1])?, zero)?;
    let low = c.where_cond(&valid, &inputs.areas.low().reshape(&[count, 1])?, zero)?;
    let radius = c.mul_scalar(&valid, 0.0)?;
    let point = a.expansion_bounds(&high, &low, &radius)?;
    let start_area = masked(a, &valid, &rows.start_area, zero)?;
    let end_area = masked(a, &valid, &rows.end_area, zero)?;
    let lower_delta = difference(a, &point, &start_area, false)?;
    let upper_delta = difference(a, &point, &end_area, true)?;
    // A candidate represents a physical state inside this band's closed support.
    let candidate = c.mul(
        &valid,
        &c.mul(
            &c.ge(lower_delta.upper(), zero)?,
            &c.ge(upper_delta.upper(), zero)?,
        )?,
    )?;
    let LocalInverse {
        candidate,
        area_above_start: lower_delta,
        area_below_end: upper_delta,
        depth_above_start: h,
        depth_below_end: e,
    } = area_query_inverse::recover(a, context, rows, &candidate, &lower_delta, &upper_delta)?;
    let missing = c.mul(&inputs.wet, &c.eq(&c.max(&candidate, &[1], false)?, zero)?)?;
    reject(
        context,
        &inputs.owners,
        &missing,
        true,
        "no band encloses the requested area point. Check profile conditioning",
    )?;
    let dry_candidate = c.mul(
        &rows.valid,
        &c.mul(
            &inputs.dry.reshape(&[count, 1])?,
            &c.mul(
                &c.eq(rows.start_depth.lower(), zero)?,
                &c.eq(rows.start_depth.upper(), zero)?,
            )?,
        )?,
    )?;
    // Dry geometry uses depth zero directly, without an inverse width quotient.
    let area_above_start = masked(a, &candidate, &lower_delta, zero)?;
    let depth_above_start = masked(a, &candidate, &h, zero)?;
    let wet_area_below_end = masked(a, &candidate, &upper_delta, zero)?;
    let wet_depth_below_end = masked(a, &candidate, &e, zero)?;
    let area_below_end = a.bounds(
        &c.where_cond(
            &dry_candidate,
            rows.end_area.lower(),
            wet_area_below_end.lower(),
        )?,
        &c.where_cond(
            &dry_candidate,
            rows.end_area.upper(),
            wet_area_below_end.upper(),
        )?,
    )?;
    let depth_below_end = a.bounds(
        &c.where_cond(
            &dry_candidate,
            rows.end_depth.lower(),
            wet_depth_below_end.lower(),
        )?,
        &c.where_cond(
            &dry_candidate,
            rows.end_depth.upper(),
            wet_depth_below_end.upper(),
        )?,
    )?;
    Ok(PrismaticAreaPointBandCoordinates {
        area_above_start,
        area_below_end,
        depth_above_start,
        depth_below_end,
        candidate: c.maximum(&candidate, &dry_candidate)?,
    })
}

pub(super) fn absolute_depth<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    rows: &BandRows<R>,
    coordinates: &PrismaticAreaPointBandCoordinates<R>,
) -> QueryResult<Interval<R>>
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
    let start = masked(a, &coordinates.candidate, &rows.start_depth, zero)?;
    let end = masked(a, &coordinates.candidate, &rows.end_depth, zero)?;
    let lower = a.add(&start, &coordinates.depth_above_start)?;
    let upper = a.sub(&end, &coordinates.depth_below_end)?;
    Ok(a.bounds(
        &c.maximum(lower.lower(), upper.lower())?,
        &c.minimum(lower.upper(), upper.upper())?,
    )?)
}

#[cfg(test)]
#[path = "area_point_inverse_tests.rs"]
mod tests;
