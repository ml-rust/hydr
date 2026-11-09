//! TwoSum stage difference and certified ordered affinity divisions without speed squares.

use crate::cpu::{
    CpuTensor,
    structures::{CpuStructureContext, pump::CpuPumpInput},
};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, TwoSumOps, UnaryOps};

pub(super) struct Normalized {
    pub high: CpuTensor,
    pub low: CpuTensor,
    pub query: CpuTensor,
    pub bound: CpuTensor,
    pub lower: CpuTensor,
    pub upper: CpuTensor,
    pub division_exact: CpuTensor,
}

pub(super) fn evaluate(
    context: &CpuStructureContext,
    donor: &CpuTensor,
    receiver: &CpuTensor,
    operation: &CpuPumpInput,
) -> Result<Normalized, StructureError> {
    let c = &context.client;
    let (high, low) = c.two_sum(receiver, &c.neg(donor)?)?;
    context.finite(&high, "original pump head difference overflows")?;
    context.finite(&low, "original pump head TwoSum witness is nonfinite")?;
    let reference = c.eq(&operation.speed, &context.one)?;
    let head_zero = c.mul(&c.eq(&high, &context.zero)?, &c.eq(&low, &context.zero)?)?;
    let generic = c.mul(
        &operation.running,
        &c.mul(
            &c.sub(&context.one, &reference)?,
            &c.sub(&context.one, &head_zero)?,
        )?,
    )?;
    let denominator = c.where_cond(&generic, &operation.speed, &context.one)?;
    let q1 = c.div(&c.where_cond(&generic, &high, &context.zero)?, &denominator)?;
    let q2 = c.div(&q1, &denominator)?;
    context.finite(&q1, "first ordered pump affinity division is nonfinite")?;
    context.finite(&q2, "second ordered pump affinity division is nonfinite")?;
    let query = c.where_cond(
        &reference,
        &high,
        &c.where_cond(&head_zero, &context.zero, &q2)?,
    )?;
    context.reject(
        &c.mul(
            &operation.running,
            &c.mul(
                &c.sub(&context.one, &head_zero)?,
                &c.eq(&query, &context.zero)?,
            )?,
        )?,
        "nonzero original pump head collapses during normalization",
    )?;
    let exact_power = c.mul(
        &operation.power_of_two,
        &c.maximum(
            &c.le(&operation.speed, &context.one)?,
            &c.ge(
                &c.abs(&q2)?,
                &c.add_scalar(&context.zero, f64::MIN_POSITIVE)?,
            )?,
        )?,
    )?;
    let exact = c.maximum(&reference, &c.maximum(&head_zero, &exact_power)?)?;
    let need_rounding = c.mul(&generic, &c.sub(&context.one, &exact_power)?)?;
    let low_abs = c.abs(&low)?;
    let low_generic = c.where_cond(&generic, &low_abs, &context.zero)?;
    let low1 = super::bounds::divide(context, &low_generic, &denominator)?;
    let low2 = super::bounds::divide(context, &low1, &denominator)?;
    let low_bound = c.where_cond(&reference, &low_abs, &low2)?;
    let first_round = c.where_cond(
        &need_rounding,
        &super::bounds::rounding(context, &q1)?,
        &context.zero,
    )?;
    let first_bound = super::bounds::divide(context, &first_round, &denominator)?;
    let second_bound = c.where_cond(
        &need_rounding,
        &super::bounds::rounding(context, &q2)?,
        &context.zero,
    )?;
    let bound = super::bounds::add(
        context,
        &low_bound,
        &super::bounds::add(context, &first_bound, &second_bound)?,
    )?;
    let (lower, upper) = super::bounds::interval(context, &query, &bound)?;
    Ok(Normalized {
        high,
        low,
        query,
        bound,
        lower,
        upper,
        division_exact: c.mul(&operation.running, &exact)?,
    })
}
