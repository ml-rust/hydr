//! Primitive nonnegative-source preparation and its unchanged filling event.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::drainage::{self, DrainageBudget};
use crate::runoff::ReservoirError;
use crate::runoff::tensor::reservoir_storage::TensorReservoirContext;

pub(super) struct Prepared<'ctx, R: Runtime<DType = DType>> {
    pub incoming_depth: Tensor<R>,
    pub hydraulic: DrainageBudget<'ctx, R>,
    pub storage_before: Tensor<R>,
    pub incoming_volume: Tensor<R>,
}

pub(super) fn prepare<'ctx, R>(
    context: &'ctx TensorReservoirContext<R>,
    old: &Tensor<R>,
    incoming: &Tensor<R>,
    dt: f64,
) -> Result<Prepared<'ctx, R>, ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    if !dt.is_finite() || dt <= 0.0 {
        return Err(ReservoirError::input(format!(
            "dt={dt}: require positive finite reservoir duration"
        )));
    }
    let c = &context.client;
    for (name, value) in [("initial depth", old), ("incoming rate", incoming)] {
        drainage::check_metadata(context, value, name)?;
        context.finite(value, name)?;
        context.reject_mask(
            &c.lt(value, &context.zero)?,
            name,
            &[("value", value)],
            &[],
            "Require nonnegative reservoir inputs",
        )?;
    }
    let incoming_depth = c.mul_scalar(incoming, dt)?;
    context.finite(&incoming_depth, "incoming interval depth")?;
    let positive_input = c.gt(incoming, &context.zero)?;
    context.reject_mask(
        &c.mul(&positive_input, &c.eq(&incoming_depth, &context.zero)?)?,
        "incoming depth",
        &[("rate", incoming)],
        &[("dt", dt)],
        "Positive input underflows. Rescale the rate or interval",
    )?;
    let available = c.add(old, &incoming_depth)?;
    context.finite(&available, "available reservoir depth")?;
    context.reject_mask(
        &c.mul(&positive_input, &c.eq(&available, old)?)?,
        "available depth",
        &[("initial", old), ("incoming depth", &incoming_depth)],
        &[],
        "Positive input is absorbed by storage rounding. Rescale storage or the interval",
    )?;
    let storage_before = drainage::volume(context, old, "initial storage")?;
    let incoming_volume = drainage::volume(context, &incoming_depth, "incoming volume")?;
    let hydraulic = drainage::prepare_budget(context, available, dt, |active| {
        let crossing = c.mul(active, &c.lt(old, &context.depression)?)?;
        let fill_depth =
            c.where_cond(&crossing, &c.sub(&context.depression, old)?, &context.zero)?;
        let fill_rate = c.where_cond(&crossing, incoming, &context.one)?;
        let fill_time = c.div(&fill_depth, &fill_rate)?;
        context.finite(&fill_time, "depression fill event time")?;
        let full_duration = c.mul_scalar(&context.one, dt)?;
        context.reject_mask(
            &c.mul(
                &crossing,
                &c.maximum(
                    &c.le(&fill_time, &context.zero)?,
                    &c.ge(&fill_time, &full_duration)?,
                )?,
            )?,
            "depression fill event",
            &[("fill time", &fill_time)],
            &[("dt", dt)],
            "Crossing time collapses or leaves the interval. Change the interval or rescale depths",
        )?;
        let duration = c.where_cond(active, &full_duration, &context.zero)?;
        let duration = c.where_cond(&crossing, &c.sub(&full_duration, &fill_time)?, &duration)?;
        context.finite(&duration, "active drainage duration")?;
        context.reject_mask(
            &c.mul(
                &crossing,
                &c.maximum(
                    &c.le(&duration, &context.zero)?,
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
    Ok(Prepared {
        incoming_depth,
        hydraulic,
        storage_before,
        incoming_volume,
    })
}
