use super::equation_ops::{scalar, scale};
use super::error::{Arithmetic, HydraulicResult};
use crate::channel::section::GatheredBands;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) struct PressureSegments<R: Runtime<DType = DType>> {
    pub(crate) area_secant: Interval<R>,
    pub(crate) integral_secant: Interval<R>,
    pub(crate) depth_increment: Interval<R>,
}

pub(crate) fn local_segments<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    low: &Interval<R>,
    high: &Interval<R>,
    zero: &Interval<R>,
) -> HydraulicResult<PressureSegments<R>>
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
    let count = low.lower().numel();
    let low_column = a.bounds(
        &low.lower().reshape(&[count, 1])?,
        &low.upper().reshape(&[count, 1])?,
    )?;
    let high_column = a.bounds(
        &high.lower().reshape(&[count, 1])?,
        &high.upper().reshape(&[count, 1])?,
    )?;
    let d1 = local_depth(a, &low_column, bands)?;
    let d2 = local_depth(a, &high_column, bands)?;
    let sum = a.add(&d1, &d2)?;
    let half_sum = scale(a, &sum, 0.5)?;
    let slope_half_sum = a.mul(&bands.slope, &half_sum)?;
    let area_secant = a.add(&bands.start_width, &slope_half_sum)?;
    let width_half_sum = a.mul(&bands.start_width, &half_sum)?;
    let base = a.add(&bands.start_area, &width_half_sum)?;
    let d1_square = a.mul(&d1, &d1)?;
    let d2_square = a.mul(&d2, &d2)?;
    let product = a.mul(&d1, &d2)?;
    let square_sum = a.add(&d1_square, &product)?;
    let square_sum = a.add(&square_sum, &d2_square)?;
    let slope_square = a.mul(&bands.slope, &square_sum)?;
    let sixth = a.div_positive(&slope_square, &Tensor::from_slice(&[6.0], &[], c.device())?)?;
    let integral_secant = a.add(&base, &sixth)?;
    let delta = a.sub(&d2, &d1)?;
    let delta = a.bounds(
        &c.maximum(delta.lower(), zero.lower())?,
        &c.maximum(delta.upper(), zero.upper())?,
    )?;
    Ok(PressureSegments {
        area_secant,
        integral_secant,
        depth_increment: delta,
    })
}

fn local_depth<R>(
    a: &mut Arithmetic<'_, R>,
    depth: &Interval<R>,
    bands: &GatheredBands<R>,
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
    let zero = scalar(a, 0.0)?;
    let local = a.sub(depth, &bands.start_depth)?;
    Ok(a.bounds(
        &c.maximum(
            &c.minimum(local.lower(), bands.thickness.lower())?,
            zero.lower(),
        )?,
        &c.maximum(
            &c.minimum(local.upper(), bands.thickness.upper())?,
            zero.upper(),
        )?,
    )?)
}
