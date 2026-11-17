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
    let (area, sound, gap) = panel_state(a, segments, lane_rows, parameter, gravity)?;
    let relative = a.div_positive_interval(&gap, &area)?;
    Ok(a.mul(&relative, &sound)?)
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
    let (area, sound, gap) = panel_state(a, segments, lane_rows, parameter, gravity)?;
    let c = a.client();
    let magnitude = c.maximum(&c.abs(gap.lower())?, &c.abs(gap.upper())?)?;
    let magnitude = a.point(&magnitude)?;
    normalized_fourth_bound(a, &sound, &magnitude, &area)
}

fn panel_state<R>(
    a: &mut Arithmetic<'_, R>,
    segments: &ConditionalBandSegments<R>,
    lane_rows: &Tensor<R>,
    parameter: &Interval<R>,
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
    let panel = segments.panel(a, lane_rows, parameter)?;
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let area = choose(a, &panel.active, &panel.area, &one)?;
    let gap = choose(a, &panel.active, &panel.gap, &zero)?;
    let hydraulic_depth = choose(a, &panel.active, &panel.hydraulic_depth, &zero)?;
    let sound = gravity_depth_root(a, gravity, &hydraulic_depth, None)?;
    Ok((area, sound, gap))
}
