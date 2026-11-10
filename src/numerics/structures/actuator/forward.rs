//! Complete-anchor endpoint enclosures and independent opening-error admission.

use super::{
    arithmetic::{difference_upper, finite, not, sum, up, used},
    motion,
};
use crate::structures::StructureError;
use crate::tensor::structures::actuator::{
    TensorActuatorAnchor, TensorActuatorContext, TensorActuatorForward, TensorActuatorPosition,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(crate) fn forward<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    anchor: &TensorActuatorAnchor<R>,
    target: &Tensor<R>,
    query: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<TensorActuatorForward<R>, StructureError>
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
    let qt = query.clone();
    let (h, l) = sum(ctx, &qt, &c.neg(&anchor.epoch)?, active)?;
    used(
        ctx,
        active,
        &c.ge(&h, &ctx.zero)?,
        "actuator query precedes its retained anchor",
    )?;
    let elapsed = c.maximum(&c.ne(&h, &ctx.zero)?, &c.ne(&l, &ctx.zero)?)?;
    let moving = c.mul(active, &elapsed)?;
    let (_, lower) = motion::endpoint(ctx, &anchor.position.lower, target, &h, &l, &moving)?;
    let (_, upper) = motion::endpoint(ctx, &anchor.position.upper, target, &h, &l, &moving)?;
    let lo = lower.lower.clone();
    let hi = upper.upper.clone();
    let exact = c.mul(&c.eq(&lo, target)?, &c.eq(&hi, target)?)?;
    let center_active = c.mul(&moving, &not(ctx, &exact)?)?;
    let (center, center_motion) =
        motion::motion(ctx, &anchor.position.center, target, &h, &l, &center_active)?;
    let center = c.where_cond(&exact, target, &center)?;
    let valid = c.mul(
        &finite(ctx, &center)?,
        &c.mul(
            &c.ge(&lo, &ctx.zero)?,
            &c.mul(
                &c.le(&lo, &center)?,
                &c.mul(&c.le(&center, &hi)?, &c.le(&hi, &ctx.maximum)?)?,
            )?,
        )?,
    )?;
    used(
        ctx,
        active,
        &valid,
        "actuator center lies outside its certified physical interval",
    )?;
    let left = difference_upper(ctx, &center, &lo, active)?;
    let right = difference_upper(ctx, &hi, &center, active)?;
    let error = c.maximum(&left, &right)?;
    let false_zero = c.mul(&c.gt(&hi, &ctx.zero)?, &c.eq(&center, &ctx.zero)?)?;
    ctx.reject(
        &c.mul(active, &false_zero)?,
        "actuator positive support cannot use represented zero opening",
    )?;
    let absolute = c.le(&error, &ctx.absolute)?;
    let need_relative = c.mul(active, &not(ctx, &absolute)?)?;
    let relative_active = c.mul(
        &need_relative,
        &c.mul(&c.gt(&ctx.relative, &ctx.zero)?, &c.gt(&center, &ctx.zero)?)?,
    )?;
    let denominator = c.where_cond(&relative_active, &c.abs(&center)?, &ctx.one)?;
    let ratio = c.div(
        &c.where_cond(&relative_active, &error, &ctx.zero)?,
        &denominator,
    )?;
    let relative = c.mul(
        &relative_active,
        &c.mul(
            &finite(ctx, &ratio)?,
            &c.le(&up(ctx, &ratio)?, &ctx.relative)?,
        )?,
    )?;
    used(
        ctx,
        active,
        &c.maximum(&absolute, &relative)?,
        "actuator opening error exceeds explicit absolute or outward-relative allowance",
    )?;
    let position = TensorActuatorPosition {
        center: c.where_cond(&exact, target, &center)?,
        lower: c.where_cond(&exact, target, &lo)?,
        upper: c.where_cond(&exact, target, &hi)?,
        error: c.where_cond(&exact, &ctx.zero, &error)?,
    };
    Ok(TensorActuatorForward {
        valid: active.clone(),
        anchor: anchor.clone(),
        query: qt,
        target: target.clone(),
        elapsed_high: h,
        elapsed_low: l,
        lower_endpoint: lower,
        upper_endpoint: upper,
        center_motion,
        position,
    })
}
