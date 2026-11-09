//! Checked authoritative storage budget with caller-owned source timing.

use super::algebra;
use crate::cpu::CpuTensor;
use crate::cpu::reservoir_storage::CpuReservoirContext;
use crate::runoff::ReservoirError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps};

/// Construction binds checked masks and storage to one immutable numerical context.
/// The budget borrows its context, preventing mutation or destruction before its last use.
/// Callers can read receipt inputs but cannot replace budget invariants.
#[derive(Debug, Clone)]
pub(crate) struct DrainageBudget<'ctx> {
    context: &'ctx CpuReservoirContext,
    available: CpuTensor,
    active: CpuTensor,
    duration: CpuTensor,
    excess: CpuTensor,
    interval: f64,
}

impl DrainageBudget<'_> {
    /// Authoritative represented storage budget, unchanged by drainage.
    pub(crate) fn available_depth(&self) -> &CpuTensor {
        &self.available
    }

    /// Checked drainage duration, zero for inactive lanes.
    pub(crate) fn active_duration(&self) -> &CpuTensor {
        &self.duration
    }

    pub(super) fn context(&self) -> &CpuReservoirContext {
        self.context
    }

    pub(super) fn active(&self) -> &CpuTensor {
        &self.active
    }

    pub(super) fn excess(&self) -> &CpuTensor {
        &self.excess
    }

    pub(super) fn interval(&self) -> f64 {
        self.interval
    }
}

/// Use the supplied budget directly, including budgets below interval-start storage.
/// The callback supplies active drainage time, without encoding sources as a rate.
pub(crate) fn prepare_budget<'ctx, F>(
    context: &'ctx CpuReservoirContext,
    available: CpuTensor,
    dt: f64,
    duration_fn: F,
) -> Result<DrainageBudget<'ctx>, ReservoirError>
where
    F: FnOnce(&CpuTensor) -> Result<CpuTensor, ReservoirError>,
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
