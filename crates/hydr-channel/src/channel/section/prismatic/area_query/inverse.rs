use super::super::{
    band_geometry::{BandRows, intersect, masked},
    context::Context,
    query_input::{Arithmetic, QueryResult},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

/// Intermediate bounds retain their arithmetic uncertainty. Apply the final candidate mask before consuming inactive lanes.
pub(in crate::channel::section::prismatic) struct LocalInverse<R: Runtime<DType = DType>> {
    pub candidate: Tensor<R>,
    pub area_above_start: Interval<R>,
    pub area_below_end: Interval<R>,
    pub depth_above_start: Interval<R>,
    pub depth_below_end: Interval<R>,
}

/// Callers prove candidate support before intersecting certified area differences with physical band bounds.
pub(in crate::channel::section::prismatic) fn recover<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    rows: &BandRows<R>,
    candidate: &Tensor<R>,
    lower_delta: &Interval<R>,
    upper_delta: &Interval<R>,
) -> QueryResult<LocalInverse<R>>
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
    let (lower_delta, candidate) = intersect(
        a,
        candidate,
        lower_delta,
        zero,
        rows.area_increment.upper(),
        zero,
    )?;
    let (upper_delta, candidate) = intersect(
        a,
        &candidate,
        upper_delta,
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
    Ok(LocalInverse {
        candidate,
        area_above_start: lower_delta,
        area_below_end: upper_delta,
        depth_above_start: h,
        depth_below_end: e,
    })
}

pub(in crate::channel::section::prismatic) fn pointed_depth<R>(
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
    let two = a.point(&Tensor::from_slice(&[2.0_f64], &[], c.device())?)?;
    let root_two = a.sqrt_nonnegative(&two)?;
    Ok(a.mul(&ratio, &root_two)?)
}
