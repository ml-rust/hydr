use super::super::{CrossSectionError, context::Context as SectionContext};
use super::compilation_bounds::{bank_slope, nonnegative, select, stack};
use super::context::{Context, arithmetic_error};
use super::inputs::{Inputs, column};
use super::topology::Topology;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{
    ArithmeticResult, Interval, IntervalContext, with_interval_arithmetic,
};
use std::sync::Arc;

struct Columns<R: Runtime<DType = DType>> {
    start_depth: Vec<Interval<R>>,
    end_depth: Vec<Interval<R>>,
    thickness: Vec<Interval<R>>,
    start_area: Vec<Interval<R>>,
    end_area: Vec<Interval<R>>,
    area_increment: Vec<Interval<R>>,
    start_width: Vec<Interval<R>>,
    slope: Vec<Interval<R>>,
    start_integral: Vec<Interval<R>>,
    pointed: Vec<Tensor<R>>,
}

impl<R: Runtime<DType = DType>> Columns<R> {
    fn new(count: usize) -> Self {
        Self {
            start_depth: Vec::with_capacity(count),
            end_depth: Vec::with_capacity(count),
            thickness: Vec::with_capacity(count),
            start_area: Vec::with_capacity(count),
            end_area: Vec::with_capacity(count),
            area_increment: Vec::with_capacity(count),
            start_width: Vec::with_capacity(count),
            slope: Vec::with_capacity(count),
            start_integral: Vec::with_capacity(count),
            pointed: Vec::with_capacity(count),
        }
    }
}

pub(super) fn compile<R>(
    source: Arc<SectionContext<R>>,
    topology: Topology,
) -> Result<Context<R>, CrossSectionError>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let (compiled, invalid, unresolved) = with_interval_arithmetic::<R, _, _, _>(
        &source.client,
        DType::F64,
        "prismatic coefficient compilation",
        |arithmetic| compile_bounds(arithmetic, Arc::clone(&source), &topology),
    )
    .map_err(arithmetic_error)?;
    if let Some(index) = source.rejected_lane(&invalid)? {
        return Err(CrossSectionError::input(
            "prismatic pressure is nonconvex within a supported band. Use static section queries for this profile",
        ).at(index));
    }
    if let Some(index) = source.rejected_lane(&unresolved)? {
        return Err(CrossSectionError::numerical(
            "prismatic span, width, or convex-pressure proof is unresolved. Review coordinate conditioning or use static section queries",
        ).at(index));
    }
    Ok(compiled)
}

fn compile_bounds<R>(
    arithmetic: &mut IntervalContext<'_, R, R::Client>,
    source: Arc<SectionContext<R>>,
    topology: &Topology,
) -> ArithmeticResult<(Context<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let client = arithmetic.client();
    let rows = topology.profiles.len();
    let inputs = Inputs::new(client, topology)?;
    let zero_tensor = Tensor::zeros(&[rows], DType::F64, client.device())?;
    let one_tensor = Tensor::ones(&[rows], DType::F64, client.device())?;
    let two_tensor = Tensor::full_scalar(&[], DType::F64, 2.0, client.device())?;
    let six_tensor = Tensor::full_scalar(&[], DType::F64, 6.0, client.device())?;
    let zero = arithmetic.point(&zero_tensor)?;
    let one = arithmetic.point(&one_tensor)?;
    let bed = arithmetic.point(&inputs.bed)?;
    let cap = arithmetic.point(&source.cap)?;
    let cap_depth = arithmetic.sub(&cap, &bed)?;
    let cap_depth = nonnegative(arithmetic, cap_depth, &zero_tensor)?;
    let left = arithmetic.point(&inputs.bottom_left)?;
    let right = arithmetic.point(&inputs.bottom_right)?;
    let bottom_width = arithmetic.sub(&right, &left)?;
    let bottom_width = nonnegative(arithmetic, bottom_width, &zero_tensor)?;
    let mut width = select(arithmetic, &inputs.pointed, &zero, &bottom_width)?;
    let mut area = arithmetic.point(&zero_tensor)?;
    let mut integral = arithmetic.point(&zero_tensor)?;
    let mut resolved = one_tensor.clone();
    let mut invalid = zero_tensor.clone();
    let mut unresolved = client.le(cap_depth.lower(), &zero_tensor)?;
    let mut columns = Columns::new(topology.maximum_bands);
    for index in 0..topology.maximum_bands {
        let valid = column(&inputs.valid, index, rows)?;
        let bottom = arithmetic.point(&column(&inputs.band_bottom, index, rows)?)?;
        let top = arithmetic.point(&column(&inputs.band_top, index, rows)?)?;
        let start_depth = if index == 0 {
            arithmetic.point(&zero_tensor)?
        } else {
            let depth = arithmetic.sub(&bottom, &bed)?;
            nonnegative(arithmetic, depth, &zero_tensor)?
        };
        let end_depth = arithmetic.sub(&top, &bed)?;
        let end_depth = nonnegative(arithmetic, end_depth, &zero_tensor)?;
        let thickness = arithmetic.sub(&top, &bottom)?;
        let thickness = select(arithmetic, &valid, &thickness, &zero)?;
        let (left_slope, left_resolved) =
            bank_slope(arithmetic, &inputs.left_bank, index, rows, &zero, &one)?;
        let (right_slope, right_resolved) =
            bank_slope(arithmetic, &inputs.right_bank, index, rows, &zero, &one)?;
        let slope = arithmetic.add(&left_slope, &right_slope)?;
        let slope = nonnegative(arithmetic, slope, &zero_tensor)?;
        let slope = select(arithmetic, &valid, &slope, &zero)?;
        let positive_thickness = client.gt(thickness.lower(), &zero_tensor)?;
        let local_resolved = client.mul(
            &client.mul(&left_resolved, &right_resolved)?,
            &positive_thickness,
        )?;
        let local_bad = client.mul(&valid, &client.eq(&local_resolved, &zero_tensor)?)?;
        unresolved = client.maximum(&unresolved, &local_bad)?;
        resolved = client.mul(
            &resolved,
            &client.where_cond(&valid, &local_resolved, &one_tensor)?,
        )?;
        let pointed = if index == 0 {
            inputs.pointed.clone()
        } else {
            zero_tensor.clone()
        };
        let width_good = client.maximum(&pointed, &client.gt(width.lower(), &zero_tensor)?)?;
        let pointed_good = client.maximum(
            &client.eq(&pointed, &zero_tensor)?,
            &client.gt(slope.lower(), &zero_tensor)?,
        )?;
        let positive_geometry = client.mul(&width_good, &pointed_good)?;
        unresolved = client.maximum(
            &unresolved,
            &client.mul(&valid, &client.eq(&positive_geometry, &zero_tensor)?)?,
        )?;
        resolved = client.mul(
            &resolved,
            &client.where_cond(&valid, &positive_geometry, &one_tensor)?,
        )?;
        let work = client.mul(&valid, &resolved)?;
        let active_width = select(arithmetic, &work, &width, &one)?;
        let active_area = select(arithmetic, &work, &area, &zero)?;
        let active_integral = select(arithmetic, &work, &integral, &zero)?;
        let active_slope = select(arithmetic, &work, &slope, &zero)?;
        let active_thickness = select(arithmetic, &work, &thickness, &zero)?;
        // C' = mT is nonnegative, so the lower band endpoint certifies each entire band.
        let width_squared = arithmetic.mul(&active_width, &active_width)?;
        let slope_area = arithmetic.mul(&active_slope, &active_area)?;
        let convexity = arithmetic.sub(&width_squared, &slope_area)?;
        let negative = client.lt(convexity.upper(), &zero_tensor)?;
        let uncertain = client.mul(
            &client.lt(convexity.lower(), &zero_tensor)?,
            &client.ge(convexity.upper(), &zero_tensor)?,
        )?;
        invalid = client.maximum(&invalid, &client.mul(&work, &negative)?)?;
        unresolved = client.maximum(&unresolved, &client.mul(&work, &uncertain)?)?;
        let d2 = arithmetic.mul(&active_thickness, &active_thickness)?;
        let d3 = arithmetic.mul(&d2, &active_thickness)?;
        let width_step = arithmetic.mul(&active_width, &active_thickness)?;
        let slope_square = arithmetic.mul(&active_slope, &d2)?;
        let half_slope_square = arithmetic.div_positive(&slope_square, &two_tensor)?;
        let area_increment = arithmetic.add(&width_step, &half_slope_square)?;
        let area_increment = nonnegative(arithmetic, area_increment, &zero_tensor)?;
        let next_area = arithmetic.add(&active_area, &area_increment)?;
        let next_area = nonnegative(arithmetic, next_area, &zero_tensor)?;
        let slope_step = arithmetic.mul(&active_slope, &active_thickness)?;
        let next_width = arithmetic.add(&active_width, &slope_step)?;
        let next_width = nonnegative(arithmetic, next_width, &zero_tensor)?;
        let area_step = arithmetic.mul(&active_area, &active_thickness)?;
        let width_square = arithmetic.mul(&active_width, &d2)?;
        let half_width_square = arithmetic.div_positive(&width_square, &two_tensor)?;
        let slope_cube = arithmetic.mul(&active_slope, &d3)?;
        let sixth_slope_cube = arithmetic.div_positive(&slope_cube, &six_tensor)?;
        let integral_area = arithmetic.add(&active_integral, &area_step)?;
        let integral_width = arithmetic.add(&integral_area, &half_width_square)?;
        let next_integral = arithmetic.add(&integral_width, &sixth_slope_cube)?;
        let next_integral = nonnegative(arithmetic, next_integral, &zero_tensor)?;
        area = select(arithmetic, &work, &next_area, &area)?;
        width = select(arithmetic, &work, &next_width, &width)?;
        integral = select(arithmetic, &work, &next_integral, &integral)?;
        columns
            .start_depth
            .push(select(arithmetic, &valid, &start_depth, &zero)?);
        columns
            .end_depth
            .push(select(arithmetic, &valid, &end_depth, &zero)?);
        columns.thickness.push(active_thickness);
        columns.start_area.push(active_area);
        columns.end_area.push(next_area);
        columns.area_increment.push(area_increment);
        columns.start_width.push(active_width);
        columns.slope.push(active_slope);
        columns.start_integral.push(active_integral);
        columns.pointed.push(client.mul(&pointed, &valid)?);
    }
    unresolved = client.maximum(&unresolved, &client.le(area.lower(), &zero_tensor)?)?;
    let pointed_columns: Vec<_> = columns.pointed.iter().collect();
    Ok((
        Context {
            source,
            maximum_bands: topology.maximum_bands,
            valid: inputs.valid.transpose(0, 1)?,
            pointed: client.stack(&pointed_columns, 1)?,
            start_depth: stack(arithmetic, &columns.start_depth)?,
            end_depth: stack(arithmetic, &columns.end_depth)?,
            thickness: stack(arithmetic, &columns.thickness)?,
            start_area: stack(arithmetic, &columns.start_area)?,
            end_area: stack(arithmetic, &columns.end_area)?,
            area_increment: stack(arithmetic, &columns.area_increment)?,
            start_width: stack(arithmetic, &columns.start_width)?,
            slope: stack(arithmetic, &columns.slope)?,
            start_integral: stack(arithmetic, &columns.start_integral)?,
            cap_depth,
            cap_area: area,
            cap_width: width,
            cap_integral: integral,
            raw_inputs: inputs,
        },
        invalid,
        unresolved,
    ))
}
