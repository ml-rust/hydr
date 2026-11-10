//! Finite outward clock endpoints with explicit positive-overflow tags.

use super::arithmetic::{down, finite, not, sum, up};
use crate::structures::StructureError;
use crate::tensor::structures::actuator::TensorActuatorContext;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) struct ClockBound<R: Runtime<DType = DType>> {
    pub value: Tensor<R>,
    pub overflow: Tensor<R>,
}

pub(super) fn clock<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    epoch: &Tensor<R>,
    duration: &Tensor<R>,
    active: &Tensor<R>,
    upper: bool,
) -> Result<ClockBound<R>, StructureError>
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
