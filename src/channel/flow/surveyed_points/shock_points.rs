use super::wave_curve_types::{CurveValue, ShockDomain, ShockOffset};
use crate::channel::flow::surveyed::{
    PrismaticRiemannRejection,
    equation_ops::reject_mask,
    error::{Arithmetic, HydraulicResult},
};
use crate::channel::section::PressureEndpoint;
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// The caller supplies matching checked source, endpoint, pressure, and gravity authority.
pub(super) fn actual<R>(
    a: &mut Arithmetic<'_, R>,
    source: &CertifiedExpansion<R>,
    target: &CertifiedExpansion<R>,
    pressure: &PressureEndpoint<R>,
) -> HydraulicResult<(CurveValue<R>, ShockOffset<R>)>
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
    let zero = Tensor::zeros(source.high().shape(), a.dtype(), c.device())?;
    let one = Tensor::ones(source.high().shape(), a.dtype(), c.device())?;
    let source_area = a.expansion_interval(source)?;
    let target_area = a.expansion_interval(target)?;
    let separation = a.expansion_interval(&pressure.delta_area)?;
    positive_domain(a, &source_area, &target_area)?;
    reject_mask(
        a,
        &c.le(separation.lower(), &zero)?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "actual shock separation lacks a strictly positive lower bound. Use the conditional wave envelope",
    )?;
    let offset = offset(a, source, target, &pressure.secant)?;
    let offset_bounds = a.expansion_interval(&offset)?;
    let physical_contraction = a.div_positive_interval(&separation, &target_area)?;
    let physical_contraction = a.bounds(
        &c.maximum(physical_contraction.lower(), &zero)?,
        &c.minimum(physical_contraction.upper(), &one)?,
    )?;

    let minimum = Tensor::full_scalar(
        source.high().shape(),
        a.dtype(),
        a.minimum_normal()?,
        c.device(),
    )?;
    let maximum = Tensor::full_scalar(
        source.high().shape(),
        a.dtype(),
        a.dtype().max_value(),
        c.device(),
    )?;
    let multiply_contraction = c.ge(physical_contraction.lower(), &minimum)?;
    let needs_gap = c.eq(&multiply_contraction, &zero)?;

    // A normal contraction avoids underflow without requiring B/A to exist in the selected arithmetic range.
    // Otherwise prove a positive normal and finite B/A before entering the gap-first expression.
    let upper_area = c.where_cond(&needs_gap, target_area.upper(), &one)?;
    let upper_area = a.point(&upper_area)?;
    let minimum_point = a.point(&minimum)?;
    let normal_threshold = a.mul(&minimum_point, &upper_area)?;
    let normal_quotient = c.mul(
        &c.gt(offset_bounds.lower(), &zero)?,
        &c.ge(offset_bounds.lower(), normal_threshold.upper())?,
    )?;

    let small_area = c.mul(&needs_gap, &c.lt(target_area.lower(), &one)?)?;
    let half = Tensor::full_scalar(source.high().shape(), a.dtype(), 0.5, c.device())?;
    // MAX*A is safe only below one. Other lanes receive a benign threshold operand before multiplication.
    let lower_area = a.point(&c.where_cond(&small_area, target_area.lower(), &half)?)?;
    let maximum_point = a.point(&maximum)?;
    let finite_threshold = a.mul(&maximum_point, &lower_area)?;
    let finite_quotient = c.maximum(
        &c.ge(target_area.lower(), &one)?,
        &c.le(offset_bounds.upper(), finite_threshold.lower())?,
    )?;
    let divide_offset = c.mul(&needs_gap, &c.mul(&normal_quotient, &finite_quotient)?)?;
    reject_mask(
        a,
        &c.eq(&c.maximum(&multiply_contraction, &divide_offset)?, &zero)?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "shock curve lacks a proved normal contraction or positive normal finite offset quotient. Refine its complete certificates",
    )?;
    let safe_offset = masked(a, &divide_offset, &offset, &one, &zero)?;
    let safe_target = masked(a, &divide_offset, target, &one, &zero)?;
    let safe_gap = masked(a, &divide_offset, &pressure.delta_area, &zero, &zero)?;
    let quotient = a.expansion_div_positive(&safe_offset, &safe_target)?;
    let gap_first = a.expansion_mul(&safe_gap, &quotient)?;
    let ratio_gap = masked(a, &multiply_contraction, &pressure.delta_area, &zero, &zero)?;
    let ratio_target = masked(a, &multiply_contraction, target, &one, &zero)?;
    let safe_contraction = a.expansion_div_positive(&ratio_gap, &ratio_target)?;
    let safe_offset = masked(a, &multiply_contraction, &offset, &one, &zero)?;
    let ratio_first = a.expansion_mul(&safe_contraction, &safe_offset)?;
    let retained = select(a, &divide_offset, &gap_first, &ratio_first)?;
    let projected = a.expansion_interval(&retained)?;
    let direct = a.mul(&physical_contraction, &offset_bounds)?;
    let physical = a.bounds(
        &c.maximum(&c.maximum(projected.lower(), direct.lower())?, &zero)?,
        &c.minimum(projected.upper(), direct.upper())?,
    )?;
    let domain = ShockDomain {
        source_area,
        target_area,
        retained_separation: a.expansion_bounds(
            pressure.delta_area.high(),
            pressure.delta_area.low(),
            pressure.delta_area.abs_error(),
        )?,
        separation,
        contraction: physical_contraction,
    };
    Ok((
        CurveValue {
            retained,
            physical,
            available: one.clone(),
        },
        ShockOffset {
            value: CurveValue {
                retained: offset,
                physical: offset_bounds,
                available: one,
            },
            domain,
        },
    ))
}

/// These certificates describe only the possible shock branch. They never authorize an actual shock speed.
pub(super) fn conditional<R>(
    a: &mut Arithmetic<'_, R>,
    source: &CertifiedExpansion<R>,
    target: &CertifiedExpansion<R>,
    pressure: &PressureEndpoint<R>,
) -> HydraulicResult<(CurveValue<R>, ShockOffset<R>)>
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
    let zero = Tensor::zeros(source.high().shape(), a.dtype(), c.device())?;
    let one = Tensor::ones(source.high().shape(), a.dtype(), c.device())?;
    let s = a.expansion_interval(source)?;
    let t = a.expansion_interval(target)?;
    let delta = a.expansion_interval(&pressure.delta_area)?;
    let source_area = a.bounds(s.lower(), &c.minimum(s.upper(), t.upper())?)?;
    let target_area = a.bounds(&c.maximum(t.lower(), s.lower())?, t.upper())?;
    let separation = a.bounds(&zero, &c.maximum(delta.upper(), &zero)?)?;
    positive_domain(a, &source_area, &target_area)?;
    let source = a.expansion_from_interval(&source_area)?;
    let target = a.expansion_from_interval(&target_area)?;
    let retained_offset = offset(a, &source, &target, &pressure.secant)?;
    let physical_offset = a.expansion_interval(&retained_offset)?;
    let contraction = a.div_positive_interval(&separation, &target_area)?;
    let contraction = a.bounds(
        &c.maximum(contraction.lower(), &zero)?,
        &c.minimum(contraction.upper(), &one)?,
    )?;
    let physical = a.mul(&contraction, &physical_offset)?;
    let physical = a.bounds(&c.maximum(physical.lower(), &zero)?, physical.upper())?;
    let retained = a.expansion_from_interval(&physical)?;
    Ok((
        CurveValue {
            retained,
            physical,
            available: one.clone(),
        },
        ShockOffset {
            value: CurveValue {
                retained: retained_offset,
                physical: physical_offset,
                available: one,
            },
            domain: ShockDomain {
                source_area,
                target_area,
                retained_separation: a.expansion_bounds(
                    pressure.delta_area.high(),
                    pressure.delta_area.low(),
                    pressure.delta_area.abs_error(),
                )?,
                separation,
                contraction,
            },
        },
    ))
}

fn positive_domain<R>(
    a: &mut Arithmetic<'_, R>,
    source: &Interval<R>,
    target: &Interval<R>,
) -> HydraulicResult<()>
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
    let zero = Tensor::zeros(source.lower().shape(), a.dtype(), c.device())?;
    let invalid = c.maximum(
        &c.maximum(&c.le(source.lower(), &zero)?, &c.le(target.lower(), &zero)?)?,
        &c.maximum(
            &c.gt(source.lower(), source.upper())?,
            &c.gt(target.lower(), target.upper())?,
        )?,
    )?;
    reject_mask(
        a,
        &invalid,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "shock domain lacks positive ordered area bounds. Refine its complete physical certificates",
    )
}

fn offset<R>(
    a: &mut Arithmetic<'_, R>,
    source: &CertifiedExpansion<R>,
    target: &CertifiedExpansion<R>,
    secant: &Interval<R>,
) -> HydraulicResult<CertifiedExpansion<R>>
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
    let zero = Tensor::zeros(source.high().shape(), a.dtype(), c.device())?;
    let one = Tensor::ones(source.high().shape(), a.dtype(), c.device())?;
    let source_bounds = a.expansion_interval(source)?;
    let target_bounds = a.expansion_interval(target)?;
    let product_first = c.mul(
        &c.lt(source_bounds.upper(), &one)?,
        &c.ge(target_bounds.upper(), &one)?,
    )?;
    let quotient_first = c.eq(&product_first, &zero)?;
    let source_root = a.expansion_sqrt_nonnegative(source)?;
    let target_root = a.expansion_sqrt_nonnegative(target)?;
    let secant_root = a.sqrt_nonnegative(secant)?;
    let secant_root = a.expansion_from_interval(&secant_root)?;
    let pk = masked(a, &product_first, &secant_root, &one, &zero)?;
    let pa = masked(a, &product_first, &target_root, &one, &zero)?;
    let ps = masked(a, &product_first, &source_root, &one, &zero)?;
    let product = a.expansion_mul(&pk, &pa)?;
    let product = a.expansion_div_positive(&product, &ps)?;
    let qk = masked(a, &quotient_first, &secant_root, &one, &zero)?;
    let qa = masked(a, &quotient_first, &target_root, &one, &zero)?;
    let qs = masked(a, &quotient_first, &source_root, &one, &zero)?;
    let quotient = a.expansion_div_positive(&qa, &qs)?;
    let quotient = a.expansion_mul(&qk, &quotient)?;
    select(a, &product_first, &product, &quotient)
}

fn masked<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
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
        &c.where_cond(mask, value.high(), fallback)?,
        &c.where_cond(mask, value.low(), zero)?,
        &c.where_cond(mask, value.abs_error(), zero)?,
    )?)
}

fn select<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    yes: &CertifiedExpansion<R>,
    no: &CertifiedExpansion<R>,
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
        &c.where_cond(mask, yes.high(), no.high())?,
        &c.where_cond(mask, yes.low(), no.low())?,
        &c.where_cond(mask, yes.abs_error(), no.abs_error())?,
    )?)
}
