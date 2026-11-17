//! Declared interval volumes remain authoritative across the soil rate bridge.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::{
    bounds,
    transfer::{self, Merged},
};
use crate::runoff::ReservoirError;
use crate::runoff::tensor::reservoir_storage::TensorReservoirContext;

#[derive(Debug, Clone)]
pub(crate) struct Bridge<R: Runtime<DType = DType>> {
    pub received_depth: Tensor<R>,
    pub received_rate: Tensor<R>,
    pub total_rate: Tensor<R>,
    pub input_depth: Tensor<R>,
    pub available_depth: Tensor<R>,
    pub rain_depth: Tensor<R>,
    pub rain_only_available: Tensor<R>,
    pub depth_increment: Tensor<R>,
    pub represented_received_volume: Tensor<R>,
    pub residual: Tensor<R>,
    pub residual_bound: Tensor<R>,
    pub uncertainty: Tensor<R>,
}

pub(crate) fn prepare<R>(
    ctx: &TensorReservoirContext<R>,
    d0: &Tensor<R>,
    rain: &Tensor<R>,
    received: &Merged<R>,
    dt: f64,
) -> Result<Bridge<R>, ReservoirError>
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
    if !dt.is_finite() || dt <= 0.0 {
        return Err(ReservoirError::input(format!(
            "dt={dt}: require positive finite transfer interval"
        )));
    }
    for (name, tensor) in [
        ("initial depth", d0),
        ("rain rate", rain),
        ("received volume", &received.volume),
        ("merge bound", &received.bound),
    ] {
        bounds::check_nonnegative(ctx, tensor, name)?;
    }
    let c = &ctx.client;
    let active = c.gt(&received.volume, &ctx.zero)?;
    ctx.reject_mask(
        &c.mul(
            &c.sub(&ctx.one, &active)?,
            &c.ne(&received.bound, &ctx.zero)?,
        )?,
        "zero received uncertainty",
        &[("received", &received.volume), ("bound", &received.bound)],
        &[],
        "Zero received volume requires exact zero merge uncertainty",
    )?;
    let received_depth = c.div(&received.volume, &ctx.area)?;
    let received_rate = c.div_scalar(&received_depth, dt)?;
    let total_rate = c.where_cond(&active, &c.add(rain, &received_rate)?, rain)?;
    let input_depth = c.mul_scalar(&total_rate, dt)?;
    let available_depth = c.add(d0, &input_depth)?;
    let rain_depth = c.mul_scalar(rain, dt)?;
    let rain_only_available = c.add(d0, &rain_depth)?;
    let depth_increment = c.where_cond(
        &active,
        &c.sub(&available_depth, &rain_only_available)?,
        &ctx.zero,
    )?;
    let represented_received_volume = c.mul(&ctx.area, &depth_increment)?;
    for (name, tensor) in [
        ("received depth", &received_depth),
        ("received rate", &received_rate),
        ("constructed incoming rate", &total_rate),
        ("constructed input depth", &input_depth),
        ("source available depth", &available_depth),
        ("rain depth", &rain_depth),
        ("rain-only available depth", &rain_only_available),
        ("represented received increment", &depth_increment),
        ("represented received volume", &represented_received_volume),
    ] {
        bounds::check_nonnegative(ctx, tensor, name)?;
    }
    let mut invalid = ctx.zero.clone();
    for mask in [
        c.le(&received_depth, &ctx.zero)?,
        c.le(&received_rate, &ctx.zero)?,
        c.le(&total_rate, rain)?,
        c.le(&input_depth, &rain_depth)?,
        c.le(&available_depth, &rain_only_available)?,
        c.le(&depth_increment, &ctx.zero)?,
        c.le(&represented_received_volume, &ctx.zero)?,
    ] {
        invalid = c.maximum(&invalid, &mask)?;
    }
    ctx.reject_mask(
        &c.mul(&active, &invalid)?,
        "positive received bridge",
        &[
            ("declared volume", &received.volume),
            ("received depth", &received_depth),
            ("received rate", &received_rate),
            ("available depth", &available_depth),
        ],
        &[("dt", dt)],
        "Positive received volume must survive every source conversion",
    )?;
    let positive_rain = c.gt(rain, &ctx.zero)?;
    ctx.reject_mask(
        &c.mul(&positive_rain, &c.eq(&rain_depth, &ctx.zero)?)?,
        "rain depth underflow",
        &[("rain rate", rain)],
        &[("dt", dt)],
        "Rescale positive rainfall or interval",
    )?;
    let positive_source = c.gt(&total_rate, &ctx.zero)?;
    ctx.reject_mask(
        &c.mul(&positive_source, &c.le(&available_depth, d0)?)?,
        "source depth absorption",
        &[("initial", d0), ("input depth", &input_depth)],
        &[],
        "Positive source must increase available storage",
    )?;

    // Mask inactive terms before outward products. Zero-transfer paths carry no charge.
    let rb = |value: &Tensor<R>| -> Result<Tensor<R>, ReservoirError> {
        bounds::positive_charge(ctx, &active, &bounds::rounding(ctx, value)?)
    };
    let interval = c.mul_scalar(&ctx.one, dt)?;
    let mut depth_bound = rb(&received_depth)?;
    for term in [
        bounds::upward_mul(ctx, &interval, &rb(&received_rate)?)?,
        bounds::upward_mul(ctx, &interval, &rb(&total_rate)?)?,
        rb(&input_depth)?,
        rb(&rain_depth)?,
        rb(&available_depth)?,
        rb(&rain_only_available)?,
        rb(&depth_increment)?,
    ] {
        depth_bound = bounds::upward_add(ctx, &depth_bound, &term)?;
    }
    let bridge_bound = bounds::upward_add(
        ctx,
        &bounds::upward_mul(ctx, &ctx.area, &depth_bound)?,
        &rb(&represented_received_volume)?,
    )?;
    let uncertainty = bounds::positive_charge(
        ctx,
        &active,
        &bounds::upward_add(ctx, &bridge_bound, &received.bound)?,
    )?;
    let residual = c.where_cond(
        &active,
        &c.sub(&represented_received_volume, &received.volume)?,
        &ctx.zero,
    )?;
    ctx.finite(&residual, "received bridge residual")?;
    let residual_bound = bounds::positive_charge(
        ctx,
        &active,
        &bounds::upward_add(ctx, &bridge_bound, &bounds::rounding(ctx, &residual)?)?,
    )?;
    ctx.reject_mask(
        &c.mul(&active, &c.gt(&c.abs(&residual)?, &residual_bound)?)?,
        "received bridge envelope",
        &[("residual", &residual), ("bound", &residual_bound)],
        &[],
        "Source bridge exceeds its arithmetic proof",
    )?;
    Ok(Bridge {
        received_depth,
        received_rate,
        total_rate,
        input_depth,
        available_depth,
        rain_depth,
        rain_only_available,
        depth_increment,
        represented_received_volume,
        residual,
        residual_bound,
        uncertainty,
    })
}

pub(crate) fn check_bridge<R>(
    ctx: &TensorReservoirContext<R>,
    d0: &Tensor<R>,
    rain: &Tensor<R>,
    received: &Merged<R>,
    dt: f64,
    receipt: &Bridge<R>,
) -> Result<(), ReservoirError>
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
    let expected = prepare(ctx, d0, rain, received, dt)?;
    for (name, actual, wanted) in [
        (
            "received depth",
            &receipt.received_depth,
            &expected.received_depth,
        ),
        (
            "received rate",
            &receipt.received_rate,
            &expected.received_rate,
        ),
        (
            "constructed incoming rate",
            &receipt.total_rate,
            &expected.total_rate,
        ),
        (
            "constructed input depth",
            &receipt.input_depth,
            &expected.input_depth,
        ),
        (
            "available source depth",
            &receipt.available_depth,
            &expected.available_depth,
        ),
        ("rain depth", &receipt.rain_depth, &expected.rain_depth),
        (
            "rain-only available depth",
            &receipt.rain_only_available,
            &expected.rain_only_available,
        ),
        (
            "received depth increment",
            &receipt.depth_increment,
            &expected.depth_increment,
        ),
        (
            "represented received volume",
            &receipt.represented_received_volume,
            &expected.represented_received_volume,
        ),
        ("bridge residual", &receipt.residual, &expected.residual),
        (
            "bridge residual bound",
            &receipt.residual_bound,
            &expected.residual_bound,
        ),
        (
            "bridge uncertainty",
            &receipt.uncertainty,
            &expected.uncertainty,
        ),
    ] {
        transfer::equal(ctx, actual, wanted, name)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
