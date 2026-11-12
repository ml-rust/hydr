use super::{
    area_point_input::{Inputs, difference},
    area_point_query::PrismaticAreaPointBandCoordinates,
    band_geometry::{BandRows, intersect, masked},
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
    let one = &context.source.one;
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
    let (lower_delta, candidate) = intersect(
        a,
        &candidate,
        &lower_delta,
        zero,
        rows.area_increment.upper(),
        zero,
    )?;
    let (upper_delta, candidate) = intersect(
        a,
        &candidate,
        &upper_delta,
        zero,
        rows.area_increment.upper(),
        zero,
    )?;
    let linear = c.mul(&candidate, &c.eq(rows.slope.upper(), zero)?)?;
    let pointed = c.mul(&candidate, &rows.pointed)?;
    let curved = c.mul(&candidate, &c.eq(&c.maximum(&linear, &pointed)?, zero)?)?;
    let delta = masked(a, &linear, &lower_delta, zero)?;
    let width = masked(a, &linear, &rows.start_width, one)?;
    let linear_depth = a.div_positive_interval(&delta, &width)?;
    let pointed_depth = if context.source.any(&pointed)? {
        let delta = masked(a, &pointed, &lower_delta, zero)?;
        let slope = masked(a, &pointed, &rows.slope, one)?;
        pointed_depth(a, &delta, &slope)?
    } else {
        a.point(&c.mul_scalar(&candidate, 0.0)?)?
    };
    let curved_depth = if context.source.any(&curved)? {
        let delta = masked(a, &curved, &lower_delta, zero)?;
        let slope = masked(a, &curved, &rows.slope, zero)?;
        let width = masked(a, &curved, &rows.start_width, one)?;
        let square = a.mul(&width, &width)?;
        let product = a.mul(&slope, &delta)?;
        let twice_product = a.add(&product, &product)?;
        let discriminant = a.add(&square, &twice_product)?;
        let root = a.sqrt_nonnegative(&discriminant)?;
        let denominator = a.add(&width, &root)?;
        let numerator = a.add(&delta, &delta)?;
        a.div_positive_interval(&numerator, &denominator)?
    } else {
        a.point(&c.mul_scalar(&candidate, 0.0)?)?
    };
    let h = a.bounds(
        &c.where_cond(
            &linear,
            linear_depth.lower(),
            &c.where_cond(&pointed, pointed_depth.lower(), curved_depth.lower())?,
        )?,
        &c.where_cond(
            &linear,
            linear_depth.upper(),
            &c.where_cond(&pointed, pointed_depth.upper(), curved_depth.upper())?,
        )?,
    )?;
    let (h, candidate) = intersect(a, &candidate, &h, zero, rows.thickness.upper(), zero)?;
    let width = masked(a, &candidate, &rows.start_width, one)?;
    let slope = masked(a, &candidate, &rows.slope, zero)?;
    let thickness = masked(a, &candidate, &rows.thickness, zero)?;
    let slope_h = a.mul(&slope, &h)?;
    let at_h = a.add(&width, &slope_h)?;
    let slope_end = a.mul(&slope, &thickness)?;
    let at_end = a.add(&width, &slope_end)?;
    let two = c.mul_scalar(one, 2.0)?;
    // Half each endpoint before adding, so the represented mean cannot overflow.
    let half_h = a.div_positive(&at_h, &two)?;
    let half_end = a.div_positive(&at_end, &two)?;
    let mean = a.add(&half_h, &half_end)?;
    let mean = masked(a, &candidate, &mean, one)?;
    let delta = masked(a, &candidate, &upper_delta, zero)?;
    let e = a.div_positive_interval(&delta, &mean)?;
    let (e, candidate) = intersect(a, &candidate, &e, zero, rows.thickness.upper(), zero)?;
    let from_upper = a.sub(&thickness, &e)?;
    let (h, candidate) = intersect(
        a,
        &candidate,
        &h,
        from_upper.lower(),
        from_upper.upper(),
        zero,
    )?;
    let from_lower = a.sub(&thickness, &h)?;
    let (e, candidate) = intersect(
        a,
        &candidate,
        &e,
        from_lower.lower(),
        from_lower.upper(),
        zero,
    )?;
    let start = masked(a, &candidate, &rows.start_depth, zero)?;
    let end = masked(a, &candidate, &rows.end_depth, zero)?;
    let h_for_depth = masked(a, &candidate, &h, zero)?;
    let e_for_depth = masked(a, &candidate, &e, zero)?;
    let from_start = a.add(&start, &h_for_depth)?;
    let from_end = a.sub(&end, &e_for_depth)?;
    let (_, candidate) = intersect(
        a,
        &candidate,
        &from_start,
        from_end.lower(),
        from_end.upper(),
        zero,
    )?;
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

pub(super) fn pointed_depth<R>(
    a: &mut Arithmetic<'_, R>,
    delta: &Interval<R>,
    slope: &Interval<R>,
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
    let root_delta = a.sqrt_nonnegative(delta)?;
    let root_slope = a.sqrt_nonnegative(slope)?;
    let ratio = a.div_positive_interval(&root_delta, &root_slope)?;
    let two = a.point(&numr::tensor::Tensor::from_slice(
        &[2.0_f64],
        &[],
        c.device(),
    )?)?;
    let root_two = a.sqrt_nonnegative(&two)?;
    Ok(a.mul(&ratio, &root_two)?)
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
