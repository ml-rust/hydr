use super::CapturedPointInput;
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::CertifiedExpansion;

pub(super) fn quotient<R>(
    a: &mut Arithmetic<'_, R>,
    wet: &Tensor<R>,
    discharge: &CertifiedExpansion<R>,
    area: &CertifiedExpansion<R>,
    constants: (&Tensor<R>, &Tensor<R>),
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
    let (zero, one) = constants;
    let numerator = masked(a, wet, discharge, zero, zero)?;
    let denominator = masked(a, wet, area, one, zero)?;
    Ok(a.expansion_div_positive(&numerator, &denominator)?)
}

pub(super) fn conditioned_difference<R>(
    a: &mut Arithmetic<'_, R>,
    active: &Tensor<R>,
    delta_discharge: &CertifiedExpansion<R>,
    delta_area: &CertifiedExpansion<R>,
    reference_velocity: &CertifiedExpansion<R>,
    denominator: &CertifiedExpansion<R>,
    constants: (&Tensor<R>, &Tensor<R>),
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
    let (zero, one) = constants;
    let delta_discharge = masked(a, active, delta_discharge, zero, zero)?;
    let delta_area = masked(a, active, delta_area, zero, zero)?;
    let reference_velocity = masked(a, active, reference_velocity, zero, zero)?;
    let denominator = masked(a, active, denominator, one, zero)?;
    let correction = a.expansion_mul(&reference_velocity, &delta_area)?;
    let numerator = a.expansion_add(&delta_discharge, &correction)?;
    Ok(a.expansion_div_positive(&numerator, &denominator)?)
}

pub(super) fn masked<R>(
    a: &mut Arithmetic<'_, R>,
    active: &Tensor<R>,
    value: &CertifiedExpansion<R>,
    fallback: &Tensor<R>,
    zero: &Tensor<R>,
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
    Ok(a.expansion_bounds(
        &c.where_cond(active, value.high(), fallback)?,
        &c.where_cond(active, value.low(), zero)?,
        &c.where_cond(active, value.abs_error(), zero)?,
    )?)
}

pub(super) fn identical<R>(
    c: &R::Client,
    left: &CertifiedExpansion<R>,
    right: &CertifiedExpansion<R>,
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
    Ok(c.mul(
        &c.eq(left.high(), right.high())?,
        &c.eq(left.low(), right.low())?,
    )?)
}

pub(super) fn same_discharge_sign<R>(
    c: &R::Client,
    input: &CapturedPointInput<R>,
    zero: &Tensor<R>,
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
    let left = &input.left_discharge;
    let right = &input.right_discharge;
    let nonnegative = |value: &CertifiedExpansion<R>| -> HydraulicResult<Tensor<R>> {
        Ok(c.maximum(
            &c.gt(value.high(), zero)?,
            &c.mul(&c.eq(value.high(), zero)?, &c.ge(value.low(), zero)?)?,
        )?)
    };
    let nonpositive = |value: &CertifiedExpansion<R>| -> HydraulicResult<Tensor<R>> {
        Ok(c.maximum(
            &c.lt(value.high(), zero)?,
            &c.mul(&c.eq(value.high(), zero)?, &c.le(value.low(), zero)?)?,
        )?)
    };
    Ok(c.maximum(
        &c.mul(&nonnegative(left)?, &nonnegative(right)?)?,
        &c.mul(&nonpositive(left)?, &nonpositive(right)?)?,
    )?)
}
