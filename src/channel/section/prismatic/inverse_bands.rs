use super::{
    band_geometry::{BandRows, hull, intersect, masked},
    compilation_bounds,
    context::Context,
    query_input::{Arithmetic, Inputs, QueryResult, reject},
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
    let one = &context.source.one;
    let count = inputs.values.numel();
    let at_cap = c.mul(
        &inputs.wet,
        &c.mul(
            &c.eq(rows.cap_area.lower(), rows.cap_area.upper())?,
            &c.eq(&inputs.values, rows.cap_area.lower())?,
        )?,
    )?;
    let active = c.mul(&inputs.wet, &c.eq(&at_cap, zero)?)?;
    let requested = a.point(&inputs.values.reshape(&[count, 1])?)?;
    let candidate = c.mul(
        &rows.valid,
        &c.mul(
            &active.reshape(&[count, 1])?,
            &c.mul(
                &c.ge(requested.lower(), rows.start_area.lower())?,
                &c.le(requested.upper(), rows.end_area.upper())?,
            )?,
        )?,
    )?;
    let requested = masked(a, &candidate, &requested, zero)?;
    let start_area = masked(a, &candidate, &rows.start_area, zero)?;
    let delta = a.sub(&requested, &start_area)?;
    let (delta, candidate) = intersect(
        a,
        &candidate,
        &delta,
        zero,
        rows.area_increment.upper(),
        zero,
    )?;
    let linear = c.mul(&candidate, &c.eq(rows.slope.upper(), zero)?)?;
    let pointed = c.mul(&candidate, &rows.pointed)?;
    let curved = c.mul(&candidate, &c.eq(&c.maximum(&linear, &pointed)?, zero)?)?;
    let linear_delta = masked(a, &linear, &delta, zero)?;
    let linear_width = masked(a, &linear, &rows.start_width, one)?;
    let linear_depth = a.div_positive_interval(&linear_delta, &linear_width)?;
    let pointed_depth = if context.source.any(&pointed)? {
        let delta = masked(a, &pointed, &delta, zero)?;
        let slope = masked(a, &pointed, &rows.slope, one)?;
        let twice_delta = a.add(&delta, &delta)?;
        let radicand = a.div_positive_interval(&twice_delta, &slope)?;
        let radicand = compilation_bounds::nonnegative(a, radicand, zero)?;
        a.sqrt_nonnegative(&radicand)?
    } else {
        a.point(&c.mul_scalar(&candidate, 0.0)?)?
    };
    let curved_depth = if context.source.any(&curved)? {
        let delta = masked(a, &curved, &delta, zero)?;
        let slope = masked(a, &curved, &rows.slope, zero)?;
        let width = masked(a, &curved, &rows.start_width, one)?;
        let square = a.mul(&width, &width)?;
        let slope_delta = a.mul(&slope, &delta)?;
        let twice_slope_delta = a.add(&slope_delta, &slope_delta)?;
        let discriminant = a.add(&square, &twice_slope_delta)?;
        let discriminant = compilation_bounds::nonnegative(a, discriminant, zero)?;
        let root = a.sqrt_nonnegative(&discriminant)?;
        let denominator = a.add(&width, &root)?;
        let numerator = a.add(&delta, &delta)?;
        a.div_positive_interval(&numerator, &denominator)?
    } else {
        a.point(&c.mul_scalar(&candidate, 0.0)?)?
    };
    let local = a.bounds(
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
    let (local, candidate) = intersect(a, &candidate, &local, zero, rows.thickness.upper(), zero)?;
    let start_depth = masked(a, &candidate, &rows.start_depth, zero)?;
    let recovered = a.add(&start_depth, &local)?;
    let low = c.maximum(rows.start_depth.lower(), zero)?;
    let high = c.minimum(
        rows.end_depth.upper(),
        &rows.cap_depth.upper().reshape(&[count, 1])?,
    )?;
    let (recovered, candidate) = intersect(a, &candidate, &recovered, &low, &high, zero)?;
    let missing = c.mul(&active, &c.eq(&c.max(&candidate, &[1], false)?, zero)?)?;
    reject(
        context,
        &inputs.owners,
        &missing,
        true,
        "no band encloses the requested area. Check profile conditioning",
    )?;
    let recovered = hull(
        a,
        &candidate,
        &recovered,
        &active,
        rows.cap_depth.upper(),
        zero,
    )?;
    Ok(a.bounds(
        &c.where_cond(&at_cap, rows.cap_depth.lower(), recovered.lower())?,
        &c.where_cond(&at_cap, rows.cap_depth.upper(), recovered.upper())?,
    )?)
}
