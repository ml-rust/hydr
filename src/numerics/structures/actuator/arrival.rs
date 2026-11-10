//! All-position arrival windows from the complete retained anchor enclosure.

use super::arithmetic::{not, sum, width};
use super::arrival_clock::clock;
use super::arrival_duration::{distance, duration};
use crate::structures::StructureError;
use crate::tensor::structures::actuator::{
    TensorActuatorAnchor, TensorActuatorArrival, TensorActuatorContext,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(crate) fn arrival<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    anchor: &TensorActuatorAnchor<R>,
    target: &Tensor<R>,
) -> Result<TensorActuatorArrival<R>, StructureError>
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
    let opening = c.lt(&anchor.position.lower, target)?;
    let closing = c.gt(&anchor.position.upper, target)?;
    let entirely_opening = c.lt(&anchor.position.upper, target)?;
    let entirely_closing = c.gt(&anchor.position.lower, target)?;
    let opening_stall = c.mul(&opening, &c.eq(&ctx.opening_rate, &ctx.zero)?)?;
    let closing_stall = c.mul(&closing, &c.eq(&ctx.closing_rate, &ctx.zero)?)?;
    let any_stall = c.maximum(&opening_stall, &closing_stall)?;
    let stalled = c.maximum(
        &c.mul(&entirely_opening, &opening_stall)?,
        &c.mul(&entirely_closing, &closing_stall)?,
    )?;
    let active = c.mul(&c.maximum(&opening, &closing)?, &not(ctx, &any_stall)?)?;

    let (horizon_high, horizon_low) = sum(
        ctx,
        &ctx.constant(ctx.schedule.end_s)?,
        &c.neg(&anchor.epoch)?,
        &active,
    )?;
    let earliest = earliest_duration(
        ctx,
        anchor,
        target,
        &entirely_opening,
        &entirely_closing,
        &active,
    )?;
    let duration_beyond = c.maximum(
        &earliest.overflow,
        &c.maximum(
            &c.gt(&earliest.value, &horizon_high)?,
            &c.mul(
                &c.eq(&earliest.value, &horizon_high)?,
                &c.lt(&horizon_low, &ctx.zero)?,
            )?,
        )?,
    )?;
    let clock_needed = c.mul(&active, &not(ctx, &duration_beyond)?)?;
    let lower = clock(ctx, &anchor.epoch, &earliest.value, &clock_needed, false)?;
    let beyond = c.mul(
        &active,
        &c.maximum(
            &duration_beyond,
            &c.maximum(
                &lower.overflow,
                &c.gt(&lower.value, &ctx.constant(ctx.schedule.end_s)?)?,
            )?,
        )?,
    )?;
    let localized = c.mul(&active, &not(ctx, &beyond)?)?;
    let upper_duration = latest_duration(ctx, anchor, target, &opening, &closing, &localized)?;
    ctx.reject(
        &upper_duration.overflow,
        "actuator in-support arrival upper duration is unrepresentable: rescale inputs",
    )?;
    let upper = clock(ctx, &anchor.epoch, &upper_duration.value, &localized, true)?;
    ctx.reject(
        &upper.overflow,
        "actuator in-support arrival upper clock is unrepresentable: rescale inputs",
    )?;
    let lower_value = c.where_cond(&localized, &lower.value, &ctx.zero)?;
    let upper_value = c.where_cond(&localized, &upper.value, &ctx.zero)?;
    ctx.reject(
        &c.mul(&localized, &c.gt(&lower_value, &upper_value)?)?,
        "actuator arrival lower exceeds its certified upper",
    )?;
    let (width_high, width_low) = width(ctx, &lower_value, &upper_value, &localized)?;
    let kind = c.where_cond(
        &any_stall,
        &c.where_cond(&stalled, &ctx.constant(3.0)?, &ctx.constant(4.0)?)?,
        &c.where_cond(&beyond, &ctx.constant(2.0)?, &localized)?,
    )?;
    Ok(TensorActuatorArrival {
        kind,
        lower: lower_value,
        upper: upper_value,
        width_high,
        width_low,
    })
}

fn earliest_duration<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    anchor: &TensorActuatorAnchor<R>,
    target: &Tensor<R>,
    entirely_opening: &Tensor<R>,
    entirely_closing: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<super::arrival_duration::DurationBound<R>, StructureError>
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
    let same_side = c.mul(active, &c.maximum(entirely_opening, entirely_closing)?)?;
    let greater = c.where_cond(entirely_opening, target, &anchor.position.lower)?;
    let lesser = c.where_cond(entirely_opening, &anchor.position.upper, target)?;
    let (lower_distance, _) = distance(ctx, &greater, &lesser, &same_side)?;
    let rate = c.where_cond(entirely_opening, &ctx.opening_rate, &ctx.closing_rate)?;
    let power = c.where_cond(entirely_opening, &ctx.opening_power, &ctx.closing_power)?;
    duration(ctx, &lower_distance, &rate, &power, &same_side, false)
}

fn latest_duration<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    anchor: &TensorActuatorAnchor<R>,
    target: &Tensor<R>,
    opening: &Tensor<R>,
    closing: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<super::arrival_duration::DurationBound<R>, StructureError>
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
    let opening_active = c.mul(active, opening)?;
    let closing_active = c.mul(active, closing)?;
    let (_, opening_distance) = distance(ctx, target, &anchor.position.lower, &opening_active)?;
    let (_, closing_distance) = distance(ctx, &anchor.position.upper, target, &closing_active)?;
    let opening_duration = duration(
        ctx,
        &opening_distance,
        &ctx.opening_rate,
        &ctx.opening_power,
        &opening_active,
        true,
    )?;
    let closing_duration = duration(
        ctx,
        &closing_distance,
        &ctx.closing_rate,
        &ctx.closing_power,
        &closing_active,
        true,
    )?;
    Ok(super::arrival_duration::DurationBound {
        value: c.maximum(&opening_duration.value, &closing_duration.value)?,
        overflow: c.maximum(&opening_duration.overflow, &closing_duration.overflow)?,
    })
}
