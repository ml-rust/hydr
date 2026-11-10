//! Finite outward clock endpoints with explicit positive-overflow tags.

use super::arithmetic::{down, finite, not, sum, up};
use crate::cpu::{CpuTensor, structures::actuator::CpuActuatorContext};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};

pub(super) struct ClockBound {
    pub value: CpuTensor,
    pub overflow: CpuTensor,
}

pub(super) fn clock(
    ctx: &CpuActuatorContext,
    epoch: &CpuTensor,
    duration: &CpuTensor,
    active: &CpuTensor,
    upper: bool,
) -> Result<ClockBound, StructureError> {
    let c = &ctx.structures.client;
    let epoch = c.where_cond(active, epoch, &ctx.zero)?;
    let duration = c.where_cond(active, duration, &ctx.zero)?;
    let ordinary = c.add(&epoch, &duration)?;
    let finite_high = finite(ctx, &ordinary)?;
    let available = c.mul(active, &finite_high)?;
    let (high, low) = sum(ctx, &epoch, &duration, &available)?;
    let value = if upper {
        c.where_cond(&c.gt(&low, &ctx.zero)?, &up(ctx, &high)?, &high)?
    } else {
        c.where_cond(
            &c.lt(&low, &ctx.zero)?,
            &c.maximum(&epoch, &down(ctx, &high)?)?,
            &high,
        )?
    };
    let finite_value = finite(ctx, &value)?;
    let overflow = c.mul(
        active,
        &c.maximum(&not(ctx, &finite_high)?, &not(ctx, &finite_value)?)?,
    )?;
    Ok(ClockBound {
        value: c.where_cond(&c.mul(&available, &finite_value)?, &value, &ctx.zero)?,
        overflow,
    })
}
