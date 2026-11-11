use super::{
    context::Context,
    query::Fields,
    query_input::{Arithmetic, QueryResult, reject},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ReduceOps, ScalarOps,
    TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(super) struct BandRows<R: Runtime<DType = DType>> {
    pub valid: Tensor<R>,
    pub pointed: Tensor<R>,
    pub start_depth: Interval<R>,
    pub end_depth: Interval<R>,
    pub thickness: Interval<R>,
    pub start_area: Interval<R>,
    pub end_area: Interval<R>,
    pub area_increment: Interval<R>,
    pub start_width: Interval<R>,
    pub slope: Interval<R>,
    pub start_integral: Interval<R>,
    pub cap_depth: Interval<R>,
    pub cap_area: Interval<R>,
    pub cap_width: Interval<R>,
    pub cap_integral: Interval<R>,
}

impl<R> BandRows<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub fn gather(
        a: &mut Arithmetic<'_, R>,
        context: &Context<R>,
        owners: &Tensor<R>,
    ) -> QueryResult<Self> {
        let c = a.client();
        let gather = |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> QueryResult<Interval<R>> {
            Ok(a.bounds(
                &c.index_select(value.lower(), 0, owners)?,
                &c.index_select(value.upper(), 0, owners)?,
            )?)
        };
        Ok(Self {
            valid: c.index_select(&context.valid, 0, owners)?,
            pointed: c.index_select(&context.pointed, 0, owners)?,
            start_depth: gather(a, &context.start_depth)?,
            end_depth: gather(a, &context.end_depth)?,
            thickness: gather(a, &context.thickness)?,
            start_area: gather(a, &context.start_area)?,
            end_area: gather(a, &context.end_area)?,
            area_increment: gather(a, &context.area_increment)?,
            start_width: gather(a, &context.start_width)?,
            slope: gather(a, &context.slope)?,
            start_integral: gather(a, &context.start_integral)?,
            cap_depth: gather(a, &context.cap_depth)?,
            cap_area: gather(a, &context.cap_area)?,
            cap_width: gather(a, &context.cap_width)?,
            cap_integral: gather(a, &context.cap_integral)?,
        })
    }
}

pub(super) fn masked<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    value: &Interval<R>,
    fallback: &Tensor<R>,
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
    Ok(a.bounds(
        &c.where_cond(mask, value.lower(), fallback)?,
        &c.where_cond(mask, value.upper(), fallback)?,
    )?)
}

pub(super) fn intersect<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    value: &Interval<R>,
    lower_limit: &Tensor<R>,
    upper_limit: &Tensor<R>,
    zero: &Tensor<R>,
) -> QueryResult<(Interval<R>, Tensor<R>)>
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
    let lower = c.maximum(value.lower(), lower_limit)?;
    let upper = c.minimum(value.upper(), upper_limit)?;
    let mask = c.mul(mask, &c.le(&lower, &upper)?)?;
    let result = a.bounds(
        &c.where_cond(&mask, &lower, zero)?,
        &c.where_cond(&mask, &upper, zero)?,
    )?;
    Ok((result, mask))
}

pub(super) fn hull<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    value: &Interval<R>,
    wet: &Tensor<R>,
    cap_upper: &Tensor<R>,
    zero: &Tensor<R>,
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
    let count = wet.numel();
    let lower = c.min(
        &c.where_cond(mask, value.lower(), &cap_upper.reshape(&[count, 1])?)?,
        &[1],
        false,
    )?;
    let upper = c.max(&c.where_cond(mask, value.upper(), zero)?, &[1], false)?;
    Ok(a.bounds(
        &c.where_cond(wet, &lower, zero)?,
        &c.where_cond(wet, &upper, zero)?,
    )?)
}

pub(super) fn empty_fields<R>(
    a: &mut Arithmetic<'_, R>,
    values: &Tensor<R>,
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
    Ok(Fields {
        depth: a.point(values)?,
        area: a.point(values)?,
        top_width: a.point(values)?,
        integral: a.point(values)?,
        hydraulic_depth: a.point(values)?,
    })
}

pub(super) fn geometry<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    rows: &BandRows<R>,
    owners: &Tensor<R>,
    depth: Interval<R>,
    wet: &Tensor<R>,
    exact_area: Option<&Tensor<R>>,
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
    let count = wet.numel();
    let column_wet = wet.reshape(&[count, 1])?;
    let query = a.bounds(
        &depth.lower().reshape(&[count, 1])?,
        &depth.upper().reshape(&[count, 1])?,
    )?;
    let candidate = c.mul(
        &rows.valid,
        &c.mul(
            &column_wet,
            &c.mul(
                &c.ge(query.upper(), rows.start_depth.lower())?,
                &c.le(query.lower(), rows.end_depth.upper())?,
            )?,
        )?,
    )?;
    let query = masked(a, &candidate, &query, zero)?;
    let start_depth = masked(a, &candidate, &rows.start_depth, zero)?;
    let local = a.sub(&query, &start_depth)?;
    let (local, candidate) = intersect(a, &candidate, &local, zero, rows.thickness.upper(), zero)?;
    let width = masked(a, &candidate, &rows.start_width, zero)?;
    let slope = masked(a, &candidate, &rows.slope, zero)?;
    let area = masked(a, &candidate, &rows.start_area, zero)?;
    let integral = masked(a, &candidate, &rows.start_integral, zero)?;
    let square = a.mul(&local, &local)?;
    let cube = a.mul(&square, &local)?;
    let slope_depth = a.mul(&slope, &local)?;
    let top_width = a.add(&width, &slope_depth)?;
    let width_depth = a.mul(&width, &local)?;
    let slope_square = a.mul(&slope, &square)?;
    let two = c.mul_scalar(&context.source.one, 2.0)?;
    let six = c.mul_scalar(&context.source.one, 6.0)?;
    let half_slope = a.div_positive(&slope_square, &two)?;
    let area_width = a.add(&area, &width_depth)?;
    let area_value = a.add(&area_width, &half_slope)?;
    let area_depth = a.mul(&area, &local)?;
    let width_square = a.mul(&width, &square)?;
    let half_width = a.div_positive(&width_square, &two)?;
    let slope_cube = a.mul(&slope, &cube)?;
    let sixth_slope = a.div_positive(&slope_cube, &six)?;
    let integral_area = a.add(&integral, &area_depth)?;
    let integral_width = a.add(&integral_area, &half_width)?;
    let integral_value = a.add(&integral_width, &sixth_slope)?;
    let (top_width, candidate) = intersect(
        a,
        &candidate,
        &top_width,
        zero,
        &rows.cap_width.upper().reshape(&[count, 1])?,
        zero,
    )?;
    let (area_value, candidate) = intersect(
        a,
        &candidate,
        &area_value,
        zero,
        &rows.cap_area.upper().reshape(&[count, 1])?,
        zero,
    )?;
    let (integral_value, candidate) = intersect(
        a,
        &candidate,
        &integral_value,
        zero,
        &rows.cap_integral.upper().reshape(&[count, 1])?,
        zero,
    )?;
    let missing = c.mul(wet, &c.eq(&c.max(&candidate, &[1], false)?, zero)?)?;
    reject(
        context,
        owners,
        &missing,
        true,
        "no band encloses this depth. Check profile conditioning",
    )?;
    let area = hull(a, &candidate, &area_value, wet, rows.cap_area.upper(), zero)?;
    let top_width = hull(a, &candidate, &top_width, wet, rows.cap_width.upper(), zero)?;
    let integral = hull(
        a,
        &candidate,
        &integral_value,
        wet,
        rows.cap_integral.upper(),
        zero,
    )?;
    reject(
        context,
        owners,
        &c.mul(wet, &c.le(top_width.lower(), zero)?)?,
        true,
        "wet width has no positive lower bound. Check profile conditioning",
    )?;
    let denominator = masked(a, wet, &top_width, &context.source.one)?;
    let numerator = if let Some(value) = exact_area {
        a.point(value)?
    } else {
        a.bounds(area.lower(), area.upper())?
    };
    let hydraulic_depth = a.div_positive_interval(&numerator, &denominator)?;
    // Nondecreasing wet width proves 0 <= A/T <= geometric depth at the physical state.
    let hydraulic_depth = a.bounds(
        &c.maximum(hydraulic_depth.lower(), zero)?,
        &c.minimum(hydraulic_depth.upper(), depth.upper())?,
    )?;
    let hydraulic_depth = masked(a, wet, &hydraulic_depth, zero)?;
    Ok(Fields {
        depth,
        area,
        top_width,
        integral,
        hydraulic_depth,
    })
}
