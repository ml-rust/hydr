//! Shape-preserving outward arithmetic and checked exact expansions.

use crate::structures::StructureError;
use crate::structures::tensor::actuator::TensorActuatorContext;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, TwoSumOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) fn finite<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::TypeConversionOps<R> + numr::ops::UnaryOps<R>,
{
    let c = &ctx.structures.client;
    Ok(c.sub(
        &ctx.one,
        &c.maximum(
            &c.cast(&c.isnan(x)?, x.dtype())?,
            &c.cast(&c.isinf(x)?, x.dtype())?,
        )?,
    )?)
}
pub(super) fn bound<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::ScalarOps<R> + numr::ops::UnaryOps<R>,
{
    Ok(crate::numerics::rounding::rounding_with_client(
        &ctx.structures.client,
        x,
    )?)
}
pub(super) fn up<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::ScalarOps<R> + numr::ops::UnaryOps<R>,
{
    Ok(ctx.structures.client.add(x, &bound(ctx, x)?)?)
}
pub(super) fn down<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::ScalarOps<R> + numr::ops::UnaryOps<R>,
{
    Ok(ctx.structures.client.sub(x, &bound(ctx, x)?)?)
}
pub(super) fn not<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>,
{
    Ok(ctx.structures.client.sub(&ctx.one, x)?)
}
pub(super) fn used<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    active: &Tensor<R>,
    valid: &Tensor<R>,
    message: &str,
) -> Result<(), StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::ReduceOps<R>,
{
    ctx.reject(
        &ctx.structures.client.mul(active, &not(ctx, valid)?)?,
        message,
    )
}
pub(super) fn sum<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &ctx.structures.client;
    let a = c.where_cond(active, a, &ctx.zero)?;
    let b = c.where_cond(active, b, &ctx.zero)?;
    let high = c.add(&a, &b)?;
    used(
        ctx,
        active,
        &finite(ctx, &high)?,
        "actuator used expansion high is nonfinite: rescale inputs",
    )?;
    let (high, low) = c.two_sum(&a, &b)?;
    used(
        ctx,
        active,
        &finite(ctx, &low)?,
        "actuator used TwoSum residual is nonfinite: rescale inputs",
    )?;
    Ok((high, low))
}
pub(super) fn outward_add<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &ctx.structures.client;
    let a = c.where_cond(active, a, &ctx.zero)?;
    let b = c.where_cond(active, b, &ctx.zero)?;
    let s = c.add(&a, &b)?;
    used(
        ctx,
        active,
        &finite(ctx, &s)?,
        "actuator uncertainty sum overflows: rescale inputs",
    )?;
    let both = c.mul(&c.ne(&a, &ctx.zero)?, &c.ne(&b, &ctx.zero)?)?;
    let value = c.where_cond(&both, &up(ctx, &s)?, &s)?;
    used(
        ctx,
        active,
        &finite(ctx, &value)?,
        "actuator outward uncertainty overflows: rescale inputs",
    )?;
    Ok(value)
}
pub(super) fn difference_upper<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
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
    let c = &ctx.structures.client;
    let (h, l) = sum(ctx, a, &c.neg(b)?, active)?;
    let padded = c.minimum(&up(ctx, &h)?, &ctx.maximum)?;
    Ok(c.where_cond(&c.gt(&l, &ctx.zero)?, &padded, &h)?)
}
pub(super) fn width<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &ctx.structures.client;
    let (h, l) = sum(ctx, upper, &c.neg(lower)?, active)?;
    let valid = c.maximum(
        &c.lt(&h, &ctx.window)?,
        &c.mul(&c.eq(&h, &ctx.window)?, &c.le(&l, &ctx.zero)?)?,
    )?;
    used(
        ctx,
        active,
        &valid,
        "actuator full arrival window exceeds explicit seconds allowance",
    )?;
    Ok((h, l))
}
pub(super) fn count<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    old: &Tensor<R>,
    increment: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::CompareOps<R> + numr::ops::ReduceOps<R>,
{
    let c = &ctx.structures.client;
    let next = c.add(old, increment)?;
    let absorbed = c.mul(&c.gt(increment, &ctx.zero)?, &c.le(&next, old)?)?;
    ctx.reject(
        &c.maximum(
            &absorbed,
            &c.gt(&next, &ctx.constant((1u64 << 53) as f64)?)?,
        )?,
        "actuator exact counter capacity exceeded",
    )?;
    Ok(next)
}
