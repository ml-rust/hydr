//! Source/event algebra shared by advancement and imported receipt checks.

use crate::cpu::CpuTensor;
use crate::cpu::reservoir_storage::CpuReservoirContext;
use crate::runoff::ReservoirError;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps};
use numr::runtime::Device;

pub(crate) const EXPONENT: f64 = 1.6666667;

pub(super) struct Prepared {
    pub incoming_depth: CpuTensor,
    pub available: CpuTensor,
    pub active: CpuTensor,
    pub duration: CpuTensor,
    pub excess: CpuTensor,
    pub storage_before: CpuTensor,
    pub incoming_volume: CpuTensor,
}

pub(super) fn prepare(
    context: &CpuReservoirContext,
    old: &CpuTensor,
    incoming: &CpuTensor,
    dt: f64,
) -> Result<Prepared, ReservoirError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(ReservoirError::input(format!(
            "dt={dt}: require positive finite reservoir duration"
        )));
    }
    let c = &context.client;
    for (name, value) in [("initial depth", old), ("incoming rate", incoming)] {
        check_metadata(context, value, name)?;
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
    let storage_before = volume(context, old, "initial storage")?;
    let incoming_volume = volume(context, &incoming_depth, "incoming volume")?;
    volume(context, &available, "available storage")?;
    let finite_roughness = c.sub(
        &context.one,
        &c.maximum(&context.instantaneous, &context.closed)?,
    )?;
    let active = c.mul(&finite_roughness, &c.gt(&available, &context.depression)?)?;
    let crossing = c.mul(&active, &c.lt(old, &context.depression)?)?;
    let fill_depth = c.where_cond(&crossing, &c.sub(&context.depression, old)?, &context.zero)?;
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
    let duration = c.where_cond(&active, &full_duration, &context.zero)?;
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
    let excess = c.where_cond(
        &active,
        &c.sub(&available, &context.depression)?,
        &context.zero,
    )?;
    context.finite(&excess, "available excess depth")?;
    context.reject_mask(
        &c.mul(&active, &c.le(&excess, &context.zero)?)?,
        "available excess",
        &[("excess", &excess)],
        &[],
        "Active drainage requires positive represented excess",
    )?;
    Ok(Prepared {
        incoming_depth,
        available,
        active,
        duration,
        excess,
        storage_before,
        incoming_volume,
    })
}

pub(super) fn check_metadata(
    context: &CpuReservoirContext,
    value: &CpuTensor,
    quantity: &str,
) -> Result<(), ReservoirError> {
    if value.shape() != context.zero.shape()
        || value.dtype() != DType::F64
        || !value.device().is_same(context.zero.device())
    {
        return Err(ReservoirError::input(format!(
            "{quantity}: require the configured CPU F64 shape {:?}",
            context.zero.shape()
        )));
    }
    Ok(())
}

pub(super) fn volume(
    context: &CpuReservoirContext,
    depth: &CpuTensor,
    quantity: &str,
) -> Result<CpuTensor, ReservoirError> {
    let c = &context.client;
    let result = c.mul(&context.area, depth)?;
    context.finite(&result, quantity)?;
    context.reject_mask(
        &c.mul(&c.gt(depth, &context.zero)?, &c.eq(&result, &context.zero)?)?,
        quantity,
        &[("value", depth), ("area", &context.area)],
        &[],
        "Positive area product underflows. Rescale geometry or state",
    )?;
    Ok(result)
}

pub(super) fn law(
    context: &CpuReservoirContext,
    excess: &CpuTensor,
    active: &CpuTensor,
) -> Result<CpuTensor, ReservoirError> {
    let c = &context.client;
    let x = c.where_cond(active, excess, &context.zero)?;
    let power = c.pow_scalar(&x, EXPONENT)?;
    context.finite(&power, "reservoir excess power")?;
    let alpha = c.where_cond(active, &context.alpha, &context.zero)?;
    let rate = c.mul(&alpha, &power)?;
    context.finite(&rate, "reservoir endpoint rate")?;
    context.reject_mask(
        &c.mul(&c.gt(&x, &context.zero)?, &c.eq(&rate, &context.zero)?)?,
        "reservoir endpoint law",
        &[("excess", &x), ("power", &power), ("coefficient", &alpha)],
        &[],
        "Positive drainage rate underflows. Rescale the reservoir",
    )?;
    Ok(rate)
}

pub(super) fn residual(
    context: &CpuReservoirContext,
    x: &CpuTensor,
    prepared: &Prepared,
) -> numr::error::Result<CpuTensor> {
    let c = &context.client;
    let x = c.where_cond(&prepared.active, x, &context.zero)?;
    let alpha = c.where_cond(&prepared.active, &context.alpha, &context.zero)?;
    let rate = c.mul(&alpha, &c.pow_scalar(&x, EXPONENT)?)?;
    let loss = c.mul(&prepared.duration, &rate)?;
    c.where_cond(
        &prepared.active,
        &c.sub(&c.add(&x, &loss)?, &prepared.excess)?,
        &context.zero,
    )
}

pub(super) fn candidate_depth(
    context: &CpuReservoirContext,
    prepared: &Prepared,
    upper: &CpuTensor,
) -> Result<CpuTensor, ReservoirError> {
    let c = &context.client;
    let algebraic = c.where_cond(
        &context.instantaneous,
        &c.minimum(&prepared.available, &context.depression)?,
        &prepared.available,
    )?;
    Ok(c.where_cond(
        &prepared.active,
        &c.add(&context.depression, upper)?,
        &algebraic,
    )?)
}
