//! Exact original-time TwoSum readiness and finite-or-overflow lateness witnesses.

use crate::structures::StructureError;
use crate::tensor::structures::control::{TensorControlContext, TensorControlTimer};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ScalarOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(crate) fn finite<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    value: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.structures.client;
    let bad = c.maximum(
        &c.cast(&c.isnan(value)?, value.dtype())?,
        &c.cast(&c.isinf(value)?, value.dtype())?,
    )?;
    Ok(c.add_scalar(&c.neg(&bad)?, 1.0)?)
}
pub(crate) fn evaluate<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    active: &Tensor<R>,
    since: &Tensor<R>,
    at: &Tensor<R>,
    delay: &Tensor<R>,
    enforce_lateness: bool,
) -> Result<TensorControlTimer<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.structures.client;
    context.reject(
        &c.mul(active, &c.gt(since, at)?)?,
        None,
        "control timer start follows its original decision: supply chronological state",
    )?;
    let started = c.where_cond(active, since, &context.zero)?;
    let evaluated = c.where_cond(active, at, &context.zero)?;
    let delay = c.where_cond(active, delay, &context.zero)?;
    let high = c.sub(&evaluated, &started)?;
    context.reject(
        &c.mul(active, &c.sub(&context.one, &finite(context, &high)?)?)?,
        None,
        "control used elapsed high is nonfinite: rescale finite timer anchors",
    )?;
    let (elapsed_high, elapsed_low) = c.two_sum(&evaluated, &c.neg(&started)?)?;
    context.reject(
        &c.mul(
            active,
            &c.sub(&context.one, &finite(context, &elapsed_low)?)?,
        )?,
        None,
        "control used elapsed residual is nonfinite: supply an admitted arithmetic envelope",
    )?;
    let ready = c.mul(
        active,
        &c.maximum(
            &c.gt(&elapsed_high, &delay)?,
            &c.mul(
                &c.eq(&elapsed_high, &delay)?,
                &c.ge(&elapsed_low, &context.zero)?,
            )?,
        )?,
    )?;
    let allowance = c.mul_scalar(active, context.timing.maximum_timer_lateness_s)?;
    let allowed_ordinary = c.add(&delay, &allowance)?;
    let overflow = c.mul(
        active,
        &c.cast(&c.isinf(&allowed_ordinary)?, allowed_ordinary.dtype())?,
    )?;
    let finite_allowed = c.mul(active, &c.sub(&context.one, &overflow)?)?;
    let (allowed_high, allowed_low) = c.two_sum(
        &c.where_cond(&finite_allowed, &delay, &context.zero)?,
        &c.where_cond(&finite_allowed, &allowance, &context.zero)?,
    )?;
    context.reject(
        &c.mul(
            &finite_allowed,
            &c.sub(
                &context.one,
                &c.mul(
                    &finite(context, &allowed_high)?,
                    &finite(context, &allowed_low)?,
                )?,
            )?,
        )?,
        None,
        "control allowed elapsed witness is nonfinite: supply admitted timing operands",
    )?;
    let lex = c.maximum(
        &c.lt(&elapsed_high, &allowed_high)?,
        &c.mul(
            &c.eq(&elapsed_high, &allowed_high)?,
            &c.le(&elapsed_low, &allowed_low)?,
        )?,
    )?;
    let within = c.mul(active, &c.maximum(&overflow, &lex)?)?;
    if enforce_lateness {
        context.reject(&c.mul(&ready,&c.sub(&context.one,&within)?)?,None,"control ready transition exceeds declared timer lateness: refine the decision schedule")?;
    }
    Ok(TensorControlTimer {
        active: active.clone(),
        started,
        evaluated,
        delay,
        elapsed_high,
        elapsed_low,
        ready,
        allowed_high,
        allowed_low,
        overflow,
        within,
    })
}
