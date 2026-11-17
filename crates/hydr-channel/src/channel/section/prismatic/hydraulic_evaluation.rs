use super::{
    band_geometry::{hull, intersect, masked},
    context::Context,
    hydraulic_geometry::{Constitutive, GatheredBands, query_error},
    query_input::reject,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ReduceOps, ScalarOps, TensorOps,
    TwoSumOps,
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    bands: &GatheredBands<R>,
    depth: &Interval<R>,
    original_area: Option<&Tensor<R>>,
) -> HydraulicResult<Constitutive<R>>
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
    let count = depth.lower().numel();
    if count == 0 {
        return Ok(Constitutive {
            depth: a.bounds(depth.lower(), depth.upper())?,
            forward_area: a.point(depth.lower())?,
            area: a.point(depth.lower())?,
            top_width: a.point(depth.lower())?,
            integral: a.point(depth.lower())?,
            hydraulic_depth: a.point(depth.lower())?,
            wet_possible: depth.lower().clone(),
            dry_possible: depth.lower().clone(),
        });
    }
    let wet = match original_area {
        Some(area) => c.gt(area, zero)?,
        None => c.gt(depth.upper(), zero)?,
    };
    let dry = match original_area {
        Some(area) => c.eq(area, zero)?,
        None => c.le(depth.lower(), zero)?,
    };
    let query = a.bounds(
        &depth.lower().reshape(&[count, 1])?,
        &depth.upper().reshape(&[count, 1])?,
    )?;
    let candidate = c.mul(
        &bands.valid,
        &c.mul(
            &wet.reshape(&[count, 1])?,
            &c.mul(
                &c.ge(query.upper(), bands.start_depth.lower())?,
                &c.le(query.lower(), bands.end_depth.upper())?,
            )?,
        )?,
    )?;
    let query = masked(a, &candidate, &query, zero).map_err(query_error)?;
    let start = masked(a, &candidate, &bands.start_depth, zero).map_err(query_error)?;
    let local = a.sub(&query, &start)?;
    let (local, candidate) = intersect(a, &candidate, &local, zero, bands.thickness.upper(), zero)
        .map_err(query_error)?;
    let width = masked(a, &candidate, &bands.start_width, zero).map_err(query_error)?;
    let slope = masked(a, &candidate, &bands.slope, zero).map_err(query_error)?;
    let start_area = masked(a, &candidate, &bands.start_area, zero).map_err(query_error)?;
    let start_integral = masked(a, &candidate, &bands.start_integral, zero).map_err(query_error)?;
    let square = a.mul(&local, &local)?;
    let cube = a.mul(&square, &local)?;
    let slope_depth = a.mul(&slope, &local)?;
    let top_width = a.add(&width, &slope_depth)?;
    let width_depth = a.mul(&width, &local)?;
    let slope_square = a.mul(&slope, &square)?;
    let two = c.mul_scalar(one, 2.0)?;
    let six = c.mul_scalar(one, 6.0)?;
    let half_slope = a.div_positive(&slope_square, &two)?;
    let area_value = a.add(&start_area, &width_depth)?;
    let area_value = a.add(&area_value, &half_slope)?;
    let area_depth = a.mul(&start_area, &local)?;
    let width_square = a.mul(&width, &square)?;
    let half_width = a.div_positive(&width_square, &two)?;
    let slope_cube = a.mul(&slope, &cube)?;
    let sixth_slope = a.div_positive(&slope_cube, &six)?;
    let integral_value = a.add(&start_integral, &area_depth)?;
    let integral_value = a.add(&integral_value, &half_width)?;
    let integral_value = a.add(&integral_value, &sixth_slope)?;
    let (top_width, candidate) = intersect(
        a,
        &candidate,
        &top_width,
        zero,
        &bands.cap_width.upper().reshape(&[count, 1])?,
        zero,
    )
    .map_err(query_error)?;
    let (area_value, candidate) = intersect(
        a,
        &candidate,
        &area_value,
        zero,
        &bands.cap_area.upper().reshape(&[count, 1])?,
        zero,
    )
    .map_err(query_error)?;
    let (integral_value, candidate) = intersect(
        a,
        &candidate,
        &integral_value,
        zero,
        &bands.cap_integral.upper().reshape(&[count, 1])?,
        zero,
    )
    .map_err(query_error)?;
    let missing = c.mul(&wet, &c.eq(&c.max(&candidate, &[1], false)?, zero)?)?;
    reject(
        context,
        &bands.section_owner,
        &missing,
        true,
        "no band encloses depth support. Check profile conditioning",
    )
    .map_err(query_error)?;

    // Pointed-bottom A/T = h/2 includes the exact dry limit without evaluating 0/0.
    let pointed = c.mul(&candidate, &bands.pointed)?;
    let ordinary = c.mul(&candidate, &c.eq(&pointed, zero)?)?;
    let ordinary_width = masked(a, &ordinary, &top_width, one).map_err(query_error)?;
    let nonpositive = c.mul(&ordinary, &c.le(ordinary_width.lower(), zero)?)?;
    let unresolved = c.max(&nonpositive, &[1], false)?;
    reject(
        context,
        &bands.section_owner,
        &unresolved,
        true,
        "wet width has no positive lower bound. Check profile conditioning",
    )
    .map_err(query_error)?;
    let numerator = match original_area {
        Some(area) => a.point(&area.reshape(&[count, 1])?)?,
        None => a.bounds(area_value.lower(), area_value.upper())?,
    };
    let numerator = masked(a, &ordinary, &numerator, zero).map_err(query_error)?;
    let ordinary_hydraulic = a.div_positive_interval(&numerator, &ordinary_width)?;
    let pointed_hydraulic = a.div_positive(&local, &two)?;
    let hydraulic = a.bounds(
        &c.where_cond(
            &pointed,
            pointed_hydraulic.lower(),
            ordinary_hydraulic.lower(),
        )?,
        &c.where_cond(
            &pointed,
            pointed_hydraulic.upper(),
            ordinary_hydraulic.upper(),
        )?,
    )?;
    let forward = hull(
        a,
        &candidate,
        &area_value,
        &wet,
        bands.cap_area.upper(),
        zero,
    )
    .map_err(query_error)?;
    let width = hull(
        a,
        &candidate,
        &top_width,
        &wet,
        bands.cap_width.upper(),
        zero,
    )
    .map_err(query_error)?;
    let integral = hull(
        a,
        &candidate,
        &integral_value,
        &wet,
        bands.cap_integral.upper(),
        zero,
    )
    .map_err(query_error)?;
    let hydraulic = hull(
        a,
        &candidate,
        &hydraulic,
        &wet,
        bands.cap_depth.upper(),
        zero,
    )
    .map_err(query_error)?;
    // Every candidate has nondecreasing width, proving 0 <= A/T <= h.
    let hydraulic = a.bounds(
        &c.maximum(hydraulic.lower(), zero)?,
        &c.minimum(hydraulic.upper(), depth.upper())?,
    )?;
    let include_dry =
        |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> HydraulicResult<Interval<R>> {
            Ok(a.bounds(&c.where_cond(&dry, zero, value.lower())?, value.upper())?)
        };
    let forward_area = include_dry(a, &forward)?;
    let area = match original_area {
        Some(area) => a.point(area)?,
        None => a.bounds(forward_area.lower(), forward_area.upper())?,
    };
    Ok(Constitutive {
        depth: a.bounds(depth.lower(), depth.upper())?,
        forward_area,
        area,
        top_width: include_dry(a, &width)?,
        integral: include_dry(a, &integral)?,
        hydraulic_depth: include_dry(a, &hydraulic)?,
        wet_possible: wet,
        dry_possible: dry,
    })
}
