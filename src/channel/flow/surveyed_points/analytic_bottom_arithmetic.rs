use super::analytic_bottom::AnalyticSource;
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) struct AnalyticOperands<'a, R: Runtime<DType = DType>> {
    pub(super) eligible: Tensor<R>,
    pub(super) rectangular: Tensor<R>,
    pub(super) pointed: Tensor<R>,
    pub(super) exact_area: &'a CertifiedExpansion<R>,
    pub(super) complete_depth: &'a Interval<R>,
    pub(super) width: &'a Interval<R>,
    pub(super) zero: &'a Tensor<R>,
    pub(super) one: &'a Tensor<R>,
    pub(super) gravity: f64,
}

pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    operands: AnalyticOperands<'_, R>,
) -> HydraulicResult<AnalyticSource<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let AnalyticOperands {
        eligible,
        rectangular,
        pointed,
        exact_area,
        complete_depth,
        width,
        zero,
        one,
        gravity,
    } = operands;
    let area = a.expansion_bounds(
        &c.where_cond(&rectangular, exact_area.high(), zero)?,
        &c.where_cond(&rectangular, exact_area.low(), zero)?,
        &c.where_cond(&rectangular, exact_area.abs_error(), zero)?,
    )?;
    let width = a.bounds(
        &c.where_cond(&rectangular, width.lower(), one)?,
        &c.where_cond(&rectangular, width.upper(), one)?,
    )?;
    let width = a.expansion_from_interval(&width)?;
    let depth = a.bounds(
        &c.where_cond(&pointed, complete_depth.lower(), zero)?,
        &c.where_cond(&pointed, complete_depth.upper(), zero)?,
    )?;
    let depth = a.expansion_from_interval(&depth)?;
    let gravity = a.expansion_point(&c.mul_scalar(one, gravity)?)?;
    let two = c.mul_scalar(one, 2.0)?;
    let four = c.mul_scalar(one, 4.0)?;
    let eight = c.mul_scalar(one, 8.0)?;
    let four_point = a.expansion_point(&four)?;
    let eight_point = a.expansion_point(&eight)?;
    let rectangular_squared = a.expansion_mul(&area, &four_point)?;
    let rectangular_squared = a.expansion_mul(&rectangular_squared, &gravity)?;
    let rectangular_squared = a.expansion_div_positive(&rectangular_squared, &width)?;
    let pointed_squared = a.expansion_mul(&depth, &eight_point)?;
    let pointed_squared = a.expansion_mul(&pointed_squared, &gravity)?;
    let squared = a.expansion_add(&rectangular_squared, &pointed_squared)?;
    let potential = a.expansion_sqrt_nonnegative(&squared)?;
    let divisor = a.expansion_point(&c.where_cond(&rectangular, &two, &four)?)?;
    let sound = a.expansion_div_positive(&potential, &divisor)?;
    let sixteen = c.mul_scalar(one, 16.0)?;
    let squared_divisor = a.expansion_point(&c.where_cond(&rectangular, &four, &sixteen)?)?;
    let sound_squared = a.expansion_div_positive(&squared, &squared_divisor)?;
    Ok(AnalyticSource {
        eligible,
        rectangular,
        pointed,
        potential,
        sound,
        sound_squared,
    })
}
