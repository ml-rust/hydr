use crate::channel::flow::surveyed::{
    equation_ops::{choose, gravity_depth_root, scalar},
    error::{Arithmetic, HydraulicResult},
    potential_difference_derivatives::normalized_fourth_bound,
};
use crate::channel::section::ConditionalBandSegments;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

/// The caller supplies checked gravity and ordinary wet segments with certified safe conditioning.
pub(crate) fn integrand<R>(
    a: &mut Arithmetic<'_, R>,
    segments: &ConditionalBandSegments<R>,
    lane_rows: &Tensor<R>,
    parameter: &Interval<R>,
    gravity: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let fields = panel_fields(a, segments, lane_rows, parameter)?;
    fields_integrand(a, &fields, gravity)
}

/// Return the existing normalized whole-panel fourth bound without discarding signed-gap uncertainty.
pub(crate) fn fourth_bound<R>(
    a: &mut Arithmetic<'_, R>,
    segments: &ConditionalBandSegments<R>,
    lane_rows: &Tensor<R>,
    parameter: &Interval<R>,
    gravity: &Interval<R>,
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
    let fields = panel_fields(a, segments, lane_rows, parameter)?;
    fields_fourth_bound(a, &fields, gravity)
}

/// Panel quantities that the integrand and the fourth bound read.
pub(super) struct PanelFields<R: Runtime<DType = DType>> {
    pub(super) active: Tensor<R>,
    pub(super) area: Interval<R>,
    pub(super) gap: Interval<R>,
    pub(super) hydraulic_depth: Interval<R>,
}

/// Hull of two endpoint states, as `ConditionalBandSegments::evaluate_range` forms it.
pub(super) fn hull<R>(
    a: &mut Arithmetic<'_, R>,
    first: &Interval<R>,
    second: &Interval<R>,
) -> HydraulicResult<Interval<R>>
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
        &c.minimum(first.lower(), second.lower())?,
        &c.maximum(first.upper(), second.upper())?,
    )?)
}

fn panel_fields<R>(
    a: &mut Arithmetic<'_, R>,
    segments: &ConditionalBandSegments<R>,
    lane_rows: &Tensor<R>,
    parameter: &Interval<R>,
) -> HydraulicResult<PanelFields<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let panel = segments.panel(a, lane_rows, parameter)?;
    Ok(PanelFields {
        active: panel.active,
        area: panel.area,
        gap: panel.gap,
        hydraulic_depth: panel.hydraulic_depth,
    })
}

pub(super) fn fields_integrand<R>(
    a: &mut Arithmetic<'_, R>,
    fields: &PanelFields<R>,
    gravity: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let (area, sound, gap) = panel_state(a, fields, gravity)?;
    let relative = a.div_positive_interval(&gap, &area)?;
    Ok(a.mul(&relative, &sound)?)
}

pub(super) fn fields_fourth_bound<R>(
    a: &mut Arithmetic<'_, R>,
    fields: &PanelFields<R>,
    gravity: &Interval<R>,
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
    let (area, sound, gap) = panel_state(a, fields, gravity)?;
    let c = a.client();
    let magnitude = c.maximum(&c.abs(gap.lower())?, &c.abs(gap.upper())?)?;
    let magnitude = a.point(&magnitude)?;
    normalized_fourth_bound(a, &sound, &magnitude, &area)
}

fn panel_state<R>(
    a: &mut Arithmetic<'_, R>,
    fields: &PanelFields<R>,
    gravity: &Interval<R>,
) -> HydraulicResult<(Interval<R>, Interval<R>, Interval<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let area = choose(a, &fields.active, &fields.area, &one)?;
    let gap = choose(a, &fields.active, &fields.gap, &zero)?;
    let hydraulic_depth = choose(a, &fields.active, &fields.hydraulic_depth, &zero)?;
    let sound = gravity_depth_root(a, gravity, &hydraulic_depth, None)?;
    Ok((area, sound, gap))
}
