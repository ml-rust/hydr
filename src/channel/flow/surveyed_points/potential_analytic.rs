use super::analytic_bottom::AnalyticSource;
use crate::channel::flow::surveyed::{
    PrismaticRiemannRejection,
    equation_ops::reject_mask,
    error::{Arithmetic, HydraulicResult},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) struct AnalyticDifference<R: Runtime<DType = DType>> {
    pub(super) eligible: Tensor<R>,
    pub(super) retained_increment: CertifiedExpansion<R>,
    pub(super) physical_increment: Interval<R>,
}

/// The caller proves common model, owner, bottom band, and gravity for both certificates and widths.
/// Eligibility masks alone do not establish that authority. Delta area retains the complete target-minus-source expression.
pub(super) fn difference<R>(
    a: &mut Arithmetic<'_, R>,
    source: &AnalyticSource<R>,
    target: &AnalyticSource<R>,
    source_width: &Interval<R>,
    target_width: &Interval<R>,
    delta_area: &CertifiedExpansion<R>,
    gravity: f64,
) -> HydraulicResult<AnalyticDifference<R>>
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
    let zero = Tensor::zeros(delta_area.high().shape(), DType::F64, c.device())?;
    let one = Tensor::ones(delta_area.high().shape(), DType::F64, c.device())?;
    let rectangular = c.mul(&source.rectangular, &target.rectangular)?;
    let pointed = c.mul(&source.pointed, &target.pointed)?;
    let eligible = c.mul(
        &c.mul(&source.eligible, &target.eligible)?,
        &c.maximum(&rectangular, &pointed)?,
    )?;
    let exact_zero = c.mul(
        &c.mul(
            &c.eq(delta_area.high(), &zero)?,
            &c.eq(delta_area.low(), &zero)?,
        )?,
        &c.eq(delta_area.abs_error(), &zero)?,
    )?;
    let active = c.mul(&eligible, &c.eq(&exact_zero, &zero)?)?;
    let rectangular = c.mul(&rectangular, &active)?;
    let pointed = c.mul(&pointed, &active)?;
    let source_phi = masked(a, &active, &source.potential, &zero, &zero)?;
    let target_phi = masked(a, &active, &target.potential, &zero, &zero)?;
    let sum = a.expansion_add(&source_phi, &target_phi)?;
    let sum = masked(a, &active, &sum, &one, &zero)?;
    let sum_bounds = a.expansion_interval(&sum)?;
    reject_mask(
        a,
        &c.mul(&active, &c.le(sum_bounds.lower(), &zero)?)?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Analytic potential difference lacks a positive potential sum. Tighten the endpoint certificates",
    )?;

    let source_t = interval_masked(a, &pointed, source_width, &zero)?;
    let target_t = interval_masked(a, &pointed, target_width, &zero)?;
    let pointed_width = a.add(&source_t, &target_t)?;
    let rectangular_source = interval_masked(a, &rectangular, source_width, &one)?;
    let rectangular_target = interval_masked(a, &rectangular, target_width, &one)?;
    let rectangular_width = a.bounds(
        &c.maximum(rectangular_source.lower(), rectangular_target.lower())?,
        &c.minimum(rectangular_source.upper(), rectangular_target.upper())?,
    )?;
    let width = a.bounds(
        &c.where_cond(&pointed, pointed_width.lower(), rectangular_width.lower())?,
        &c.where_cond(&pointed, pointed_width.upper(), rectangular_width.upper())?,
    )?;
    reject_mask(
        a,
        &c.mul(&active, &c.le(width.lower(), &zero)?)?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Analytic potential difference lacks a positive width denominator. Tighten the endpoint certificates",
    )?;
    let width = a.expansion_from_interval(&width)?;
    let delta = masked(a, &active, delta_area, &zero, &zero)?;
    let numerator = a.expansion_div_positive(&delta, &width)?;
    let gravity = a.expansion_point(&c.mul_scalar(&one, gravity)?)?;
    let numerator = a.expansion_mul(&numerator, &gravity)?;
    let four = c.mul_scalar(&one, 4.0)?;
    let sixteen = c.mul_scalar(&one, 16.0)?;
    let factor = a.expansion_point(&c.where_cond(&pointed, &sixteen, &four)?)?;
    let numerator = a.expansion_mul(&numerator, &factor)?;
    let retained_increment = a.expansion_div_positive(&numerator, &sum)?;
    let projected = a.expansion_interval(&retained_increment)?;
    let numerator = a.expansion_interval(&numerator)?;
    let direct = a.div_positive_interval(&numerator, &sum_bounds)?;
    let delta = a.expansion_interval(&delta)?;
    let lower = c.maximum(projected.lower(), direct.lower())?;
    let upper = c.minimum(projected.upper(), direct.upper())?;
    let physical_increment = a.bounds(
        &c.where_cond(
            &c.ge(delta.lower(), &zero)?,
            &c.maximum(&lower, &zero)?,
            &lower,
        )?,
        &c.where_cond(
            &c.le(delta.upper(), &zero)?,
            &c.minimum(&upper, &zero)?,
            &upper,
        )?,
    )?;
    Ok(AnalyticDifference {
        eligible,
        retained_increment,
        physical_increment,
    })
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

fn interval_masked<R>(
    a: &mut Arithmetic<'_, R>,
    active: &Tensor<R>,
    value: &Interval<R>,
    fallback: &Tensor<R>,
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
        &c.where_cond(active, value.lower(), fallback)?,
        &c.where_cond(active, value.upper(), fallback)?,
    )?)
}
