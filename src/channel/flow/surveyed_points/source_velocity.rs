use super::types::CapturedPointInput;
use crate::channel::flow::surveyed::equation_ops::choose_expansion;
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult, checked};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::CertifiedExpansion;
use std::sync::Arc;

#[path = "source_velocity_selection.rs"]
mod selection;

/// Physical quotient certificates bound to their captured conserved inputs.
pub(crate) struct SourceVelocity<R: Runtime<DType = DType>> {
    input: Arc<CapturedPointInput<R>>,
    pub(crate) left: CertifiedExpansion<R>,
    pub(crate) right: CertifiedExpansion<R>,
    pub(crate) difference: CertifiedExpansion<R>,
    /// One only where both physical source velocities are defined.
    pub(crate) both_wet: Tensor<R>,
    /// Wet rows with exactly identical normalized area and discharge points.
    pub(crate) uniform: Tensor<R>,
}

impl<R: Runtime<DType = DType>> SourceVelocity<R> {
    pub(crate) fn input(&self) -> &CapturedPointInput<R> {
        &self.input
    }
}

pub(crate) fn prepare<R>(
    client: &R::Client,
    input: &Arc<CapturedPointInput<R>>,
) -> HydraulicResult<SourceVelocity<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    checked::<R, _>(client, "surveyed point source velocities", |a| {
        let zero = Tensor::zeros(input.owners.shape(), a.dtype(), client.device())?;
        let one = Tensor::ones(input.owners.shape(), a.dtype(), client.device())?;
        let left_wet = client.eq(&input.left_dry, &zero)?;
        let right_wet = client.eq(&input.right_dry, &zero)?;
        let both_wet = client.mul(&left_wet, &right_wet)?;
        let left = quotient(
            a,
            &left_wet,
            &input.left_discharge,
            &input.left_area,
            (&zero, &one),
        )?;
        let right = quotient(
            a,
            &right_wet,
            &input.right_discharge,
            &input.right_area,
            (&zero, &one),
        )?;
        let equal_area = identical(client, &input.left_area, &input.right_area)?;
        let equal_discharge = identical(client, &input.left_discharge, &input.right_discharge)?;
        let uniform = client.mul(&both_wet, &client.mul(&equal_area, &equal_discharge)?)?;

        let same_sign = same_discharge_sign(client, input, &zero)?;
        let eligible = client.mul(
            &both_wet,
            &client.mul(&same_sign, &client.eq(&uniform, &zero)?)?,
        )?;
        let left_area = masked(a, &eligible, &input.left_area, &one, &zero)?;
        let right_area = masked(a, &eligible, &input.right_area, &one, &zero)?;
        let delta_area = a.expansion_sub(&left_area, &right_area)?;
        let difference_bounds = a.expansion_interval(&delta_area)?;
        let left_order = client.ge(difference_bounds.lower(), &zero)?;
        let right_order = client.le(difference_bounds.upper(), &zero)?;

        // These differences prove comparability without first doubling a possibly large area.
        let left_change = masked(a, &left_order, &delta_area, &zero, &zero)?;
        let left_other = masked(a, &left_order, &right_area, &one, &zero)?;
        let left_margin = a.expansion_sub(&left_other, &left_change)?;
        let left_margin = a.expansion_interval(&left_margin)?;
        let left_mask = client.mul(
            &eligible,
            &client.mul(&left_order, &client.ge(left_margin.lower(), &zero)?)?,
        )?;
        let right_change = masked(a, &right_order, &delta_area, &zero, &zero)?;
        let right_other = masked(a, &right_order, &left_area, &one, &zero)?;
        let right_margin = a.expansion_add(&right_other, &right_change)?;
        let right_margin = a.expansion_interval(&right_margin)?;
        let right_mask = client.mul(
            &eligible,
            &client.mul(
                &client.eq(&left_mask, &zero)?,
                &client.mul(&right_order, &client.ge(right_margin.lower(), &zero)?)?,
            )?,
        )?;
        let conditioned = client.maximum(&left_mask, &right_mask)?;
        let fallback = client.mul(
            &both_wet,
            &client.mul(
                &client.eq(&uniform, &zero)?,
                &client.eq(&conditioned, &zero)?,
            )?,
        )?;

        // Only certified conditioned rows enter the difference-first numerator.
        let left_discharge = masked(a, &conditioned, &input.left_discharge, &zero, &zero)?;
        let right_discharge = masked(a, &conditioned, &input.right_discharge, &zero, &zero)?;
        let delta_discharge = a.expansion_sub(&right_discharge, &left_discharge)?;
        let left_difference = conditioned_difference(
            a,
            &left_mask,
            &delta_discharge,
            &delta_area,
            &left,
            &input.right_area,
            (&zero, &one),
        )?;
        let right_difference = conditioned_difference(
            a,
            &right_mask,
            &delta_discharge,
            &delta_area,
            &right,
            &input.left_area,
            (&zero, &one),
        )?;
        let fallback_left = masked(a, &fallback, &left, &zero, &zero)?;
        let fallback_right = masked(a, &fallback, &right, &zero, &zero)?;
        let independent = a.expansion_sub(&fallback_right, &fallback_left)?;
        let difference = choose_expansion(a, &right_mask, &right_difference, &independent)?;
        let difference = choose_expansion(a, &left_mask, &left_difference, &difference)?;
        Ok(SourceVelocity {
            input: Arc::clone(input),
            left,
            right,
            difference,
            both_wet,
            uniform,
        })
    })
}

fn quotient<R>(
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

fn conditioned_difference<R>(
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

fn masked<R>(
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

fn identical<R>(
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

fn same_discharge_sign<R>(
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
