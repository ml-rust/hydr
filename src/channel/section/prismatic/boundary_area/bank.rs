use super::super::inputs::{BankInputs, column};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;

pub(super) fn station<R>(
    a: &mut Arithmetic<'_, R>,
    bank: &BankInputs<R>,
    left: bool,
    band: usize,
    stage: &Tensor<R>,
    valid: &Tensor<R>,
) -> HydraulicResult<CertifiedExpansion<R>>
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
    let rows = stage.numel();
    let zero = Tensor::zeros(stage.shape(), DType::F64, c.device())?;
    let one = Tensor::ones(stage.shape(), DType::F64, c.device())?;
    let low_elevation = column(&bank.low_elevation, band, rows)?;
    let high_elevation = column(&bank.high_elevation, band, rows)?;
    let vertical = column(&bank.vertical, band, rows)?;
    let low_station = column(
        if left {
            &bank.right_station
        } else {
            &bank.left_station
        },
        band,
        rows,
    )?;
    let high_station = column(
        if left {
            &bank.left_station
        } else {
            &bank.right_station
        },
        band,
        rows,
    )?;
    let at_low = c.eq(stage, &low_elevation)?;
    let at_high = c.eq(stage, &high_elevation)?;
    let interpolate = c.mul(
        valid,
        &c.mul(
            &c.eq(&vertical, &zero)?,
            &c.mul(&c.eq(&at_low, &zero)?, &c.eq(&at_high, &zero)?)?,
        )?,
    )?;

    // Mask before eager arithmetic, including unused original endpoint differences.
    let low_z = a.expansion_point(&c.where_cond(&interpolate, &low_elevation, &zero)?)?;
    let high_z = a.expansion_point(&c.where_cond(&interpolate, &high_elevation, &one)?)?;
    let active_stage = a.expansion_point(&c.where_cond(&interpolate, stage, &zero)?)?;
    let low_x = a.expansion_point(&c.where_cond(&interpolate, &low_station, &zero)?)?;
    let high_x = a.expansion_point(&c.where_cond(&interpolate, &high_station, &zero)?)?;
    let rise = a.expansion_sub(&active_stage, &low_z)?;
    let height = a.expansion_sub(&high_z, &low_z)?;
    let fraction = a.expansion_div_positive(&rise, &height)?;
    let run = a.expansion_sub(&high_x, &low_x)?;
    let displacement = a.expansion_mul(&run, &fraction)?;
    let interpolated = a.expansion_add(&low_x, &displacement)?;
    let endpoint = c.where_cond(&at_high, &high_station, &low_station)?;
    let high = c.where_cond(
        valid,
        &c.where_cond(&interpolate, interpolated.high(), &endpoint)?,
        &zero,
    )?;
    let low = c.where_cond(&interpolate, interpolated.low(), &zero)?;
    let radius = c.where_cond(&interpolate, interpolated.abs_error(), &zero)?;
    Ok(a.expansion_bounds(&high, &low, &radius)?)
}
