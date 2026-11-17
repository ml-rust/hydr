use super::analytic_bottom::AnalyticSource;
use crate::channel::flow::surveyed::{
    PrismaticRiemannRejection,
    equation_ops::reject_mask,
    error::{Arithmetic, HydraulicResult},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) struct AnalyticSonic<R: Runtime<DType = DType>> {
    pub(super) eligible: Tensor<R>,
    pub(super) retained_residual: CertifiedExpansion<R>,
    pub(super) physical_residual: Interval<R>,
}

/// The caller proves common model, owner, bottom band, and gravity for the target and outgoing invariant.
/// Epsilon is positive except for structural dry equality. Matching masks or overlapping certificates prove no authority.
pub(super) fn residual<R>(
    a: &mut Arithmetic<'_, R>,
    target: &AnalyticSource<R>,
    epsilon: &CertifiedExpansion<R>,
) -> HydraulicResult<AnalyticSonic<R>>
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
    let zero = Tensor::zeros(epsilon.high().shape(), a.dtype(), c.device())?;
    let one = Tensor::ones(epsilon.high().shape(), a.dtype(), c.device())?;
    let eligible = c.mul(
        &target.eligible,
        &c.maximum(&target.rectangular, &target.pointed)?,
    )?;
    let epsilon_zero = exact_zero::<R>(a, epsilon, &zero)?;
    let sound_zero = exact_zero::<R>(a, &target.sound_squared, &zero)?;
    let structural_zero = c.mul(&epsilon_zero, &sound_zero)?;
    let active = c.mul(&eligible, &c.eq(&structural_zero, &zero)?)?;
    let epsilon = masked(a, &active, epsilon, &zero, &zero)?;
    let epsilon_bounds = a.expansion_interval(&epsilon)?;
    reject_mask(
        a,
        &c.mul(&active, &c.le(epsilon_bounds.lower(), &zero)?)?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Analytic sonic invariant lacks a positive lower bound. Tighten its complete source certificate",
    )?;
    let sound = masked(a, &active, &target.sound, &zero, &zero)?;
    let squared = masked(a, &active, &target.sound_squared, &zero, &zero)?;
    let three = c.mul_scalar(&one, 3.0)?;
    let five = c.mul_scalar(&one, 5.0)?;
    let nine = c.mul_scalar(&one, 9.0)?;
    let twenty_five = c.mul_scalar(&one, 25.0)?;
    let k = a.expansion_point(&c.where_cond(&target.pointed, &five, &three)?)?;
    let k_squared = a.expansion_point(&c.where_cond(&target.pointed, &twenty_five, &nine)?)?;
    let scaled_squared = a.expansion_mul(&k_squared, &squared)?;
    let epsilon_squared = a.expansion_mul(&epsilon, &epsilon)?;
    let numerator = a.expansion_sub(&scaled_squared, &epsilon_squared)?;
    let scaled_sound = a.expansion_mul(&k, &sound)?;
    let denominator = a.expansion_add(&scaled_sound, &epsilon)?;
    let denominator = masked(a, &active, &denominator, &one, &zero)?;
    let denominator_bounds = a.expansion_interval(&denominator)?;
    reject_mask(
        a,
        &c.mul(&active, &c.le(denominator_bounds.lower(), &zero)?)?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Analytic sonic denominator lacks a positive lower bound. Tighten the complete target and invariant certificates",
    )?;
    let retained_residual = a.expansion_div_positive(&numerator, &denominator)?;
    let projected = a.expansion_interval(&retained_residual)?;
    let numerator = a.expansion_interval(&numerator)?;
    let direct = a.div_positive_interval(&numerator, &denominator_bounds)?;
    let lower = c.maximum(projected.lower(), direct.lower())?;
    let upper = c.minimum(projected.upper(), direct.upper())?;
    let physical_residual = a.bounds(
        &c.where_cond(
            &c.ge(numerator.lower(), &zero)?,
            &c.maximum(&lower, &zero)?,
            &lower,
        )?,
        &c.where_cond(
            &c.le(numerator.upper(), &zero)?,
            &c.minimum(&upper, &zero)?,
            &upper,
        )?,
    )?;
    Ok(AnalyticSonic {
        eligible,
        retained_residual,
        physical_residual,
    })
}

fn exact_zero<R>(
    a: &Arithmetic<'_, R>,
    value: &CertifiedExpansion<R>,
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
    let c = a.client();
    Ok(c.mul(
        &c.mul(&c.eq(value.high(), zero)?, &c.eq(value.low(), zero)?)?,
        &c.eq(value.abs_error(), zero)?,
    )?)
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
