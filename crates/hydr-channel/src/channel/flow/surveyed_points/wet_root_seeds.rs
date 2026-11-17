use super::{
    paired_sources::{PairedSources, SelectedRows},
    potential_range::select_expansion,
    wave_curve_types::CurveValue,
};
use crate::channel::{
    flow::surveyed::{
        analytic_bottom,
        error::{Arithmetic, HydraulicError, HydraulicResult, checked},
    },
    section::SupportedAreaQuery,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::ArithmeticError;

/// Choose represented query coordinates. Only later residual signs establish root existence.
pub(super) fn lower_seed<R>(
    parent: &PairedSources<'_, R>,
    selected: &SelectedRows<R>,
    chart: &SupportedAreaQuery<R>,
    f0: &CurveValue<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = parent.geometry.client();
    let (mut seed, epsilon, width, slope, rectangular, pointed) =
        checked::<R, _>(c, "wet-root lower seed operands", |a| {
            let zero = Tensor::zeros(chart.query.parameter.lower().shape(), a.dtype(), c.device())?;
            let input = selected.velocity.input();
            let left = a.expansion_interval(&input.left_area)?;
            let right = a.expansion_interval(&input.right_area)?;
            let ordinary = c.maximum(left.lower(), right.lower())?;
            let ordinary = c.minimum(&ordinary, chart.query.parameter.upper())?;
            let positive = c.gt(&ordinary, &zero)?;
            let difference = a.expansion_interval(&selected.velocity.difference)?;
            let use_ordinary = c.mul(&positive, &c.gt(difference.upper(), &zero)?)?;
            let seed = c.where_cond(&use_ordinary, &ordinary, chart.query.parameter.upper())?;
            let seed = clip(a, chart, &seed)?;

            let zero_expansion = a.expansion_point(&zero)?;
            let epsilon = a.expansion_sub(&zero_expansion, &f0.retained)?;
            let epsilon = a.expansion_interval(&epsilon)?;
            let positive = c.mul(&f0.available, &c.gt(epsilon.lower(), &zero)?)?;
            let bands = &selected.bands;
            let width = a.bounds(
                &column(bands.start_width.lower())?,
                &column(bands.start_width.upper())?,
            )?;
            let slope = a.bounds(&column(bands.slope.lower())?, &column(bands.slope.upper())?)?;
            let (rectangular, pointed) = analytic_bottom::kinds(
                a,
                &column(&bands.valid)?,
                &column(&bands.pointed)?,
                &width,
                &slope,
            )?;
            Ok((
                seed,
                epsilon.lower().clone(),
                width.lower().clone(),
                slope.lower().clone(),
                c.mul(&rectangular, &positive)?,
                c.mul(&pointed, &positive)?,
            ))
        })?;

    for (mask, factor, pointed) in [(&rectangular, &width, false), (&pointed, &slope, true)] {
        let rows = c.nonzero(mask)?.flatten()?;
        if rows.numel() == 0 {
            continue;
        }
        let factor = c.index_select(factor, 0, &rows)?;
        let epsilon = c.index_select(&epsilon, 0, &rows)?;
        let candidate = estimate(parent, &factor, &epsilon, pointed)?;
        if let Some(candidate) = candidate {
            seed = checked::<R, _>(c, "wet-root analytic seed region", |a| {
                let compact_chart = chart.select(a, &rows)?;
                let candidate = clip(a, &compact_chart, &candidate)?;
                let point = a.expansion_point(&candidate)?;
                let input = selected.velocity.input();
                let left = select_expansion(a, &input.left_area, &rows)?;
                let right = select_expansion(a, &input.right_area, &rows)?;
                let left_difference = a.expansion_sub(&point, &left)?;
                let right_difference = a.expansion_sub(&point, &right)?;
                let left_difference = a.expansion_interval(&left_difference)?;
                let right_difference = a.expansion_interval(&right_difference)?;
                let end = c.index_select(&column(selected.bands.end_area.lower())?, 0, &rows)?;
                let zero = Tensor::zeros(rows.shape(), a.dtype(), c.device())?;
                let eligible = c.mul(
                    &c.le(&candidate, &end)?,
                    &c.mul(
                        &c.le(left_difference.upper(), &zero)?,
                        &c.le(right_difference.upper(), &zero)?,
                    )?,
                )?;
                let accepted = c.nonzero(&eligible)?.flatten()?;
                let owners = c.index_select(&rows, 0, &accepted)?;
                let values = c.index_select(&candidate, 0, &accepted)?;
                Ok(c.index_put(&seed, 0, &owners, &values)?)
            })?;
        }
    }
    Ok(seed)
}

/// Choose deficits from retained capacity differences, without subtracting projected areas.
pub(super) fn cap_seed<R>(
    parent: &PairedSources<'_, R>,
    selected: &SelectedRows<R>,
    chart: &SupportedAreaQuery<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    checked::<R, _>(
        parent.geometry.client(),
        "wet-root retained cap seed",
        |a| {
            let c = a.client();
            let capacity = selected
                .boundaries
                .boundary(a, selected.boundaries.cap_indices())?;
            let input = selected.velocity.input();
            let left = a.expansion_sub(&capacity, &input.left_area)?;
            let right = a.expansion_sub(&capacity, &input.right_area)?;
            let left = a.expansion_interval(&left)?;
            let right = a.expansion_interval(&right)?;
            let candidate = c.maximum(left.lower(), right.lower())?;
            let zero = Tensor::zeros(candidate.shape(), a.dtype(), c.device())?;
            let positive = c.gt(&candidate, &zero)?;
            let candidate = c.where_cond(&positive, &candidate, chart.query.parameter.upper())?;
            clip(a, chart, &candidate)
        },
    )
}

fn estimate<R>(
    parent: &PairedSources<'_, R>,
    factor: &Tensor<R>,
    epsilon: &Tensor<R>,
    pointed: bool,
) -> HydraulicResult<Option<Tensor<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    // Isolate optional range exhaustion so no rejected arithmetic context is reused.
    let result = checked::<R, _>(
        parent.geometry.client(),
        "wet-root analytic seed ratio",
        |a| {
            let gravity = Tensor::full_scalar(
                &[],
                a.dtype(),
                parent.evaluation.config.gravity_m_per_s2,
                a.client().device(),
            )?;
            let coefficient = Tensor::full_scalar(
                &[],
                a.dtype(),
                if pointed { 2048.0 } else { 16.0 },
                a.client().device(),
            )?;
            if pointed {
                Ok(a.positive_product_ratio_lower(
                    &[factor, epsilon, epsilon, epsilon, epsilon],
                    &[&coefficient, &gravity, &gravity],
                )?)
            } else {
                Ok(a.positive_product_ratio_lower(
                    &[factor, epsilon, epsilon],
                    &[&coefficient, &gravity],
                )?)
            }
        },
    );
    match result {
        Ok(candidate) => Ok(Some(candidate)),
        Err(HydraulicError::Arithmetic(ArithmeticError::Range { .. })) => Ok(None),
        Err(error) => Err(error),
    }
}

fn column<R: Runtime<DType = DType>>(value: &Tensor<R>) -> HydraulicResult<Tensor<R>> {
    Ok(value.narrow(1, 0, 1)?.squeeze(Some(1)))
}

fn clip<R>(
    a: &Arithmetic<'_, R>,
    chart: &SupportedAreaQuery<R>,
    candidate: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(a.client().maximum(
        chart.query.parameter.lower(),
        &a.client()
            .minimum(candidate, chart.query.parameter.upper())?,
    )?)
}
