use super::equation_ops::{scalar, scale, select_interval};
use super::error::{Arithmetic, HydraulicResult};
use super::potential_difference_lanes::DifferenceLanes;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) fn integrand<R>(
    a: &mut Arithmetic<'_, R>,
    lanes: &DifferenceLanes<R>,
    owner: &Tensor<R>,
    s: &Interval<R>,
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
    let (area, sound, delta) = panel_state(a, lanes, owner, s)?;
    let relative = a.div_positive_interval(&delta, &area)?;
    Ok(a.mul(&relative, &sound)?)
}

pub(crate) fn panel_fourth_bound<R>(
    a: &mut Arithmetic<'_, R>,
    lanes: &DifferenceLanes<R>,
    owner: &Tensor<R>,
    s: &Interval<R>,
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
    let (area, sound, delta) = panel_state(a, lanes, owner, s)?;
    let c = a.client();
    let absolute = c.maximum(&c.abs(delta.lower())?, &c.abs(delta.upper())?)?;
    let absolute = a.point(&absolute)?;
    normalized_fourth_bound(a, &sound, &absolute, &area)
}

pub(crate) fn normalized_fourth_bound<R>(
    a: &mut Arithmetic<'_, R>,
    sound: &Interval<R>,
    absolute_delta: &Interval<R>,
    area: &Interval<R>,
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
    // Certified convex pressure gives C >= m*A and m >= 0, hence 2*m*D/C <= 2*D/A.
    // Leibniz then bounds |G''''| by (255/4)*c*(D/A)^5 throughout this panel.
    let relative = a.div_positive_interval(absolute_delta, area)?;
    let mut bound = a.mul(sound, &relative)?;
    for _ in 0..4 {
        bound = a.mul(&bound, &relative)?;
    }
    let bound = scale(a, &bound, 255.0 / 4.0)?;
    Ok(bound.upper().clone())
}

fn panel_state<R>(
    a: &mut Arithmetic<'_, R>,
    lanes: &DifferenceLanes<R>,
    owner: &Tensor<R>,
    s: &Interval<R>,
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
    let c = a.client();
    let original = select_interval(a, &lanes.original, owner)?;
    let delta = select_interval(a, &lanes.delta, owner)?;
    let hull = select_interval(a, &lanes.hull, owner)?;
    let offset = a.mul(&delta, s)?;
    let affine = a.add(&original, &offset)?;
    // Every complete physical segment lies inside the endpoint hull, for either delta sign.
    let area = a.bounds(
        &c.maximum(affine.lower(), hull.lower())?,
        &c.minimum(affine.upper(), hull.upper())?,
    )?;
    let start = select_interval(a, &lanes.start_area, owner)?;
    let width = select_interval(a, &lanes.width, owner)?;
    let slope = select_interval(a, &lanes.slope, owner)?;
    let zero = scalar(a, 0.0)?;
    let local = a.sub(&area, &start)?;
    let local = a.bounds(&c.maximum(local.lower(), zero.lower())?, local.upper())?;
    let width_squared = a.mul(&width, &width)?;
    let variable = a.mul(&slope, &local)?;
    let variable = scale(a, &variable, 2.0)?;
    let squared_width = a.add(&width_squared, &variable)?;
    let top_width = a.sqrt_nonnegative(&squared_width)?;
    let depth = a.div_positive_interval(&area, &top_width)?;
    let squared_sound = a.mul(&lanes.gravity, &depth)?;
    let sound = a.sqrt_nonnegative(&squared_sound)?;
    Ok((area, sound, delta))
}
