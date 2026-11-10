//! Source-specific filling time with an authoritative post-loss storage budget.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::sources::Sources;
use crate::numerics::reservoir::drainage::{self, DrainageBudget};
use crate::runoff::ReservoirError;
use crate::tensor::reservoir_storage::TensorReservoirContext;

pub(super) fn prepare<'ctx, R>(
    ctx: &'ctx TensorReservoirContext<R>,
    sources: &Sources<R>,
    dt: f64,
) -> Result<(DrainageBudget<'ctx, R>, Tensor<R>), ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>,
{
    let c = &ctx.client;
    let mut timing_rate = ctx.zero.clone();
    let budget = drainage::prepare_budget(ctx, sources.available_depth.clone(), dt, |active| {
        let crossing = c.mul(active, &c.lt(&sources.supply.old_depth, &ctx.depression)?)?;
        let net_crossing = c.mul(&crossing, &c.sub(&ctx.one, &sources.direct_timing)?)?;
        let delta = c.where_cond(
            &net_crossing,
            &c.sub(&sources.available_depth, &sources.supply.old_depth)?,
            &ctx.zero,
        )?;
        let net_rate = c.div_scalar(&delta, dt)?;
        ctx.finite(&net_rate, "net crossing rate")?;
        timing_rate = c.where_cond(&sources.direct_timing, &sources.supply.rain_rate, &net_rate)?;
        ctx.reject_mask(
            &c.mul(&crossing, &c.le(&timing_rate, &ctx.zero)?)?,
            "positive filling rate",
            &[("rate", &timing_rate)],
            &[("dt", dt)],
            "Crossing requires a positive represented timing rate",
        )?;
        let fill_depth = c.where_cond(
            &crossing,
            &c.sub(&ctx.depression, &sources.supply.old_depth)?,
            &ctx.zero,
        )?;
        let fill_rate = c.where_cond(&crossing, &timing_rate, &ctx.one)?;
        let fill_time = c.div(&fill_depth, &fill_rate)?;
        ctx.finite(&fill_time, "depression fill event time")?;
        let full_duration = c.mul_scalar(&ctx.one, dt)?;
        ctx.reject_mask(
            &c.mul(
                &crossing,
                &c.maximum(
                    &c.le(&fill_time, &ctx.zero)?,
                    &c.ge(&fill_time, &full_duration)?,
                )?,
            )?,
            "depression fill event",
            &[("fill time", &fill_time)],
            &[("dt", dt)],
            "Crossing time collapses or leaves the interval. Change the interval or rescale depths",
        )?;
        let duration = c.where_cond(active, &full_duration, &ctx.zero)?;
        let duration = c.where_cond(&crossing, &c.sub(&full_duration, &fill_time)?, &duration)?;
        ctx.finite(&duration, "active drainage duration")?;
        ctx.reject_mask(
            &c.mul(
                &crossing,
                &c.maximum(
                    &c.le(&duration, &ctx.zero)?,
                    &c.ge(&duration, &full_duration)?,
                )?,
            )?,
            "active drainage event",
            &[("duration", &duration)],
            &[("dt", dt)],
            "Drainage time collapses or leaves the interval. Change the interval or rescale depths",
        )?;
        Ok(duration)
    })?;
    Ok((budget, timing_rate))
}
