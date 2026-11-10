//! Outward travel durations from exact represented distance bounds.

use super::arithmetic::{down, finite, not, sum, up};
use crate::cpu::{CpuTensor, structures::actuator::CpuActuatorContext};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, UnaryOps};

pub(super) struct DurationBound {
    pub value: CpuTensor,
    pub overflow: CpuTensor,
}

pub(super) fn distance(
    ctx: &CpuActuatorContext,
    greater: &CpuTensor,
    lesser: &CpuTensor,
    active: &CpuTensor,
) -> Result<(CpuTensor, CpuTensor), StructureError> {
    let c = &ctx.structures.client;
    let (high, low) = sum(ctx, greater, &c.neg(lesser)?, active)?;
    let lower = c.where_cond(
        &c.lt(&low, &ctx.zero)?,
        &c.maximum(&ctx.zero, &down(ctx, &high)?)?,
        &high,
    )?;
    // Physical range bounds distance even when outward padding overflows.
    let upper = c.where_cond(
        &c.gt(&low, &ctx.zero)?,
        &c.minimum(&ctx.maximum, &up(ctx, &high)?)?,
        &high,
    )?;
    Ok((lower, upper))
}

pub(super) fn duration(
    ctx: &CpuActuatorContext,
    distance: &CpuTensor,
    rate: &CpuTensor,
    power: &CpuTensor,
    active: &CpuTensor,
    upper: bool,
) -> Result<DurationBound, StructureError> {
    let c = &ctx.structures.client;
    let positive = c.mul(active, &c.gt(distance, &ctx.zero)?)?;
    let distance = c.where_cond(&positive, distance, &ctx.zero)?;
    let rate = c.where_cond(&positive, rate, &ctx.one)?;
    let quotient = c.div(&distance, &rate)?;
    let finite_quotient = finite(ctx, &quotient)?;
    let available = c.mul(&positive, &finite_quotient)?;
    let safe_quotient = c.where_cond(&available, &quotient, &ctx.zero)?;
    let exact = exact_quotient(ctx, &distance, &rate, power, &safe_quotient, &available)?;
    let padded = if upper {
        up(ctx, &safe_quotient)?
    } else {
        c.maximum(&ctx.zero, &down(ctx, &safe_quotient)?)?
    };
    let selected = c.where_cond(&exact, &safe_quotient, &padded)?;
    let finite_selected = finite(ctx, &selected)?;
    let overflow = c.mul(
        &positive,
        &c.maximum(&not(ctx, &finite_quotient)?, &not(ctx, &finite_selected)?)?,
    )?;
    let valid = c.mul(&available, &finite_selected)?;
    Ok(DurationBound {
        value: c.where_cond(&valid, &selected, &ctx.zero)?,
        overflow,
    })
}

fn exact_quotient(
    ctx: &CpuActuatorContext,
    distance: &CpuTensor,
    rate: &CpuTensor,
    power: &CpuTensor,
    quotient: &CpuTensor,
    active: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    let c = &ctx.structures.client;
    let candidate = c.mul(&c.mul(active, power)?, &c.gt(quotient, &ctx.zero)?)?;
    let rate_at_most_one = c.le(rate, &ctx.one)?;
    let quotient_normal = c.ge(quotient, &ctx.constant(f64::MIN_POSITIVE)?)?;
    let direct = c.maximum(&rate_at_most_one, &quotient_normal)?;
    let back_needed = c.mul(&candidate, &not(ctx, &direct)?)?;
    let back = c.mul(
        &c.where_cond(&back_needed, quotient, &ctx.zero)?,
        &c.where_cond(&back_needed, rate, &ctx.one)?,
    )?;
    let back_exact = c.mul(&finite(ctx, &back)?, &c.eq(&back, distance)?)?;
    Ok(c.mul(&candidate, &c.maximum(&direct, &back_exact)?)?)
}
