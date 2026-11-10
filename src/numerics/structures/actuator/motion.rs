//! Signed-high/low fused motion and finite physical endpoint images.

use super::{
    arithmetic::{down, finite, not, outward_add, up, used},
    fused,
};
use crate::structures::StructureError;
use crate::tensor::structures::actuator::{
    TensorActuatorContext, TensorActuatorEndpoint, TensorActuatorMotion,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, TwoSumOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) fn motion<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
    target: &Tensor<R>,
    h: &Tensor<R>,
    l: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<(Tensor<R>, TensorActuatorMotion<R>), StructureError>
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
    let opening = c.lt(x, target)?;
    let rate = c.where_cond(&opening, &ctx.opening_rate, &ctx.closing_rate)?;
    let power = c.where_cond(&opening, &ctx.opening_power, &ctx.closing_power)?;
    let moving = c.mul(active, &c.mul(&c.ne(x, target)?, &c.gt(&rate, &ctx.zero)?)?)?;
    let k = c.where_cond(&opening, &rate, &c.neg(&rate)?)?;
    let primary = fused::stage(ctx, &k, h, x, &power, &moving)?;
    let second = c.mul(&moving, &c.ne(l, &ctx.zero)?)?;
    let low = fused::stage(ctx, &k, l, &primary.result, &power, &second)?;
    let s = c.where_cond(&second, &low.result, &primary.result)?;
    let combined = outward_add(ctx, &primary.error, &low.error, &second)?;
    let error = c.where_cond(&second, &combined, &primary.error)?;
    let (lower_valid, lower) = raw(ctx, &s, &error, &moving, false)?;
    let (upper_valid, upper) = raw(ctx, &s, &error, &moving, true)?;
    let bad_open = c.mul(&opening, &c.mul(&upper_valid, &c.lt(&upper, x)?)?)?;
    let bad_close = c.mul(
        &not(ctx, &opening)?,
        &c.mul(&lower_valid, &c.gt(&lower, x)?)?,
    )?;
    ctx.reject(
        &c.mul(&moving, &c.maximum(&bad_open, &bad_close)?)?,
        "actuator raw motion enclosure contradicts directional support",
    )?;
    let center = c.where_cond(&opening, &c.minimum(&s, target)?, &c.maximum(&s, target)?)?;
    Ok((
        c.where_cond(&moving, &center, x)?,
        TensorActuatorMotion {
            valid: moving,
            direction: opening,
            rate,
            primary,
            low,
            error,
            raw_lower_valid: lower_valid,
            raw_lower: lower,
            raw_upper_valid: upper_valid,
            raw_upper: upper,
        },
    ))
}
fn raw<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    s: &Tensor<R>,
    error: &Tensor<R>,
    active: &Tensor<R>,
    upper: bool,
) -> Result<(Tensor<R>, Tensor<R>), StructureError>
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
    let signed = if upper { error.clone() } else { c.neg(error)? };
    let ordinary = c.add(s, &signed)?;
    let ordinary_finite = finite(ctx, &ordinary)?;
    let witness = c.mul(active, &ordinary_finite)?;
    let (high, low) = c.two_sum(
        &c.where_cond(&witness, s, &ctx.zero)?,
        &c.where_cond(&witness, &signed, &ctx.zero)?,
    )?;
    used(
        ctx,
        &witness,
        &finite(ctx, &low)?,
        "actuator raw position TwoSum residual is nonfinite: rescale inputs",
    )?;
    let padded = if upper {
        up(ctx, &high)?
    } else {
        down(ctx, &high)?
    };
    let pad = if upper {
        c.gt(&low, &ctx.zero)?
    } else {
        c.lt(&low, &ctx.zero)?
    };
    let raw = c.where_cond(&pad, &padded, &high)?;
    let valid = c.mul(&ordinary_finite, &finite(ctx, &raw)?)?;
    Ok((valid.clone(), c.where_cond(&valid, &raw, &ctx.zero)?))
}
pub(super) fn endpoint<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
    target: &Tensor<R>,
    h: &Tensor<R>,
    l: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<(Tensor<R>, TensorActuatorEndpoint<R>), StructureError>
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
    let (center, motion) = motion(ctx, x, target, h, l, active)?;
    let physical_low = c.minimum(x, target)?;
    let physical_high = c.maximum(x, target)?;
    let lo = c.where_cond(
        &motion.raw_lower_valid,
        &c.maximum(
            &physical_low,
            &c.minimum(&physical_high, &motion.raw_lower)?,
        )?,
        &physical_low,
    )?;
    let hi = c.where_cond(
        &motion.raw_upper_valid,
        &c.maximum(
            &physical_low,
            &c.minimum(&physical_high, &motion.raw_upper)?,
        )?,
        &physical_high,
    )?;
    Ok((
        center,
        TensorActuatorEndpoint {
            anchor: x.clone(),
            lower: c.where_cond(&motion.valid, &lo, x)?,
            upper: c.where_cond(&motion.valid, &hi, x)?,
            motion,
        },
    ))
}
