//! Checked authoritative storage budget with caller-owned source timing.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::algebra;
use crate::runoff::ReservoirError;
use crate::runoff::tensor::reservoir_storage::TensorReservoirContext;

/// Construction binds checked masks and storage to one immutable numerical context.
/// The budget borrows its context, preventing mutation or destruction before its last use.
/// Callers can read receipt inputs but cannot replace budget invariants.
#[derive(Debug, Clone)]
pub(crate) struct DrainageBudget<'ctx, R: Runtime<DType = DType>> {
    context: &'ctx TensorReservoirContext<R>,
    available: Tensor<R>,
    active: Tensor<R>,
    duration: Tensor<R>,
    excess: Tensor<R>,
    interval: f64,
}

impl<R> DrainageBudget<'_, R>
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
    /// Authoritative represented storage budget, unchanged by drainage.
    pub(crate) fn available_depth(&self) -> &Tensor<R> {
        &self.available
    }

    /// Checked drainage duration, zero for inactive lanes.
    pub(crate) fn active_duration(&self) -> &Tensor<R> {
        &self.duration
    }

    pub(super) fn context(&self) -> &TensorReservoirContext<R> {
        self.context
    }

    pub(super) fn active(&self) -> &Tensor<R> {
        &self.active
    }

    pub(super) fn excess(&self) -> &Tensor<R> {
        &self.excess
    }

    pub(super) fn interval(&self) -> f64 {
        self.interval
    }
}

/// Use the supplied budget directly, including budgets below interval-start storage.
/// The callback supplies active drainage time, without encoding sources as a rate.
pub(crate) fn prepare_budget<'ctx, R, F>(
    context: &'ctx TensorReservoirContext<R>,
    available: Tensor<R>,
    dt: f64,
    duration_fn: F,
) -> Result<DrainageBudget<'ctx, R>, ReservoirError>
where
    F: FnOnce(&Tensor<R>) -> Result<Tensor<R>, ReservoirError>,
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
            "dt={dt}: require positive finite drainage duration"
        )));
    }
    algebra::check_metadata(context, &available, "provided available depth")?;
    context.finite(&available, "provided available depth")?;
    let c = &context.client;
    context.reject_mask(
        &c.lt(&available, &context.zero)?,
        "provided available depth",
        &[("available", &available)],
        &[],
        "Supply a nonnegative authoritative storage budget",
    )?;
    algebra::volume(context, &available, "available storage")?;
    let finite_roughness = c.sub(
        &context.one,
        &c.maximum(&context.instantaneous, &context.closed)?,
    )?;
    let active = c.mul(&finite_roughness, &c.gt(&available, &context.depression)?)?;
    let duration = duration_fn(&active)?;
    algebra::check_metadata(context, &duration, "provided active duration")?;
    context.finite(&duration, "provided active duration")?;
    let full_duration = c.mul_scalar(&context.one, dt)?;
    let invalid_active = c.mul(
        &active,
        &c.maximum(
            &c.le(&duration, &context.zero)?,
            &c.gt(&duration, &full_duration)?,
        )?,
    )?;
    let invalid_inactive = c.mul(
        &c.sub(&context.one, &active)?,
        &c.ne(&duration, &context.zero)?,
    )?;
    context.reject_mask(
        &c.maximum(&invalid_active, &invalid_inactive)?,
        "provided drainage duration",
        &[("duration", &duration), ("active", &active)],
        &[("dt", dt)],
        "Require 0<duration<=dt for drainage and exact zero duration otherwise",
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
    Ok(DrainageBudget {
        context,
        available,
        active,
        duration,
        excess,
        interval: dt,
    })
}
