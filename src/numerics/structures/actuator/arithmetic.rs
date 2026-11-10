//! Shape-preserving outward arithmetic and checked exact expansions.

use crate::cpu::{CpuTensor, structures::actuator::CpuActuatorContext};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, TwoSumOps, TypeConversionOps, UnaryOps};

pub(super) fn finite(ctx: &CpuActuatorContext, x: &CpuTensor) -> Result<CpuTensor, StructureError> {
    let c = &ctx.structures.client;
    Ok(c.sub(
        &ctx.one,
        &c.maximum(
            &c.cast(&c.isnan(x)?, x.dtype())?,
            &c.cast(&c.isinf(x)?, x.dtype())?,
        )?,
    )?)
}
pub(super) fn bound(ctx: &CpuActuatorContext, x: &CpuTensor) -> Result<CpuTensor, StructureError> {
    Ok(crate::numerics::subcatchment::bounds::rounding_with_client(
        &ctx.structures.client,
        x,
    )?)
}
pub(super) fn up(ctx: &CpuActuatorContext, x: &CpuTensor) -> Result<CpuTensor, StructureError> {
    Ok(ctx.structures.client.add(x, &bound(ctx, x)?)?)
}
pub(super) fn down(ctx: &CpuActuatorContext, x: &CpuTensor) -> Result<CpuTensor, StructureError> {
    Ok(ctx.structures.client.sub(x, &bound(ctx, x)?)?)
}
pub(super) fn not(ctx: &CpuActuatorContext, x: &CpuTensor) -> Result<CpuTensor, StructureError> {
    Ok(ctx.structures.client.sub(&ctx.one, x)?)
}
pub(super) fn used(
    ctx: &CpuActuatorContext,
    active: &CpuTensor,
    valid: &CpuTensor,
    message: &str,
) -> Result<(), StructureError> {
    ctx.reject(
        &ctx.structures.client.mul(active, &not(ctx, valid)?)?,
        message,
    )
}
pub(super) fn sum(
    ctx: &CpuActuatorContext,
    a: &CpuTensor,
    b: &CpuTensor,
    active: &CpuTensor,
) -> Result<(CpuTensor, CpuTensor), StructureError> {
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
pub(super) fn outward_add(
    ctx: &CpuActuatorContext,
    a: &CpuTensor,
    b: &CpuTensor,
    active: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
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
pub(super) fn difference_upper(
    ctx: &CpuActuatorContext,
    a: &CpuTensor,
    b: &CpuTensor,
    active: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    let c = &ctx.structures.client;
    let (h, l) = sum(ctx, a, &c.neg(b)?, active)?;
    let padded = c.minimum(&up(ctx, &h)?, &ctx.maximum)?;
    Ok(c.where_cond(&c.gt(&l, &ctx.zero)?, &padded, &h)?)
}
pub(super) fn width(
    ctx: &CpuActuatorContext,
    lower: &CpuTensor,
    upper: &CpuTensor,
    active: &CpuTensor,
) -> Result<(CpuTensor, CpuTensor), StructureError> {
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
pub(super) fn count(
    ctx: &CpuActuatorContext,
    old: &CpuTensor,
    increment: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
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
