//! Represented reservoir law, geometry products, and root reconstruction.

use super::budget::DrainageBudget;
use crate::cpu::CpuTensor;
use crate::cpu::reservoir_storage::CpuReservoirContext;
use crate::runoff::ReservoirError;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps};
use numr::runtime::Device;

pub(crate) const EXPONENT: f64 = 1.6666667;

pub(in crate::numerics::reservoir) fn check_metadata(
    context: &CpuReservoirContext,
    value: &CpuTensor,
    quantity: &str,
) -> Result<(), ReservoirError> {
    if value.shape() != context.zero.shape()
        || value.dtype() != DType::F64
        || !value.device().is_same(context.zero.device())
    {
        return Err(ReservoirError::input(format!(
            "{quantity}: shape={:?}, dtype={:?}: require the configured CPU F64 shape {:?} on the matching device",
            value.shape(),
            value.dtype(),
            context.zero.shape()
        )));
    }
    Ok(())
}

pub(in crate::numerics::reservoir) fn volume(
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
    prepared: &DrainageBudget<'_>,
) -> numr::error::Result<CpuTensor> {
    let c = &context.client;
    let x = c.where_cond(prepared.active(), x, &context.zero)?;
    let alpha = c.where_cond(prepared.active(), &context.alpha, &context.zero)?;
    let rate = c.mul(&alpha, &c.pow_scalar(&x, EXPONENT)?)?;
    let loss = c.mul(prepared.active_duration(), &rate)?;
    c.where_cond(
        prepared.active(),
        &c.sub(&c.add(&x, &loss)?, prepared.excess())?,
        &context.zero,
    )
}

pub(super) fn candidate_depth(
    context: &CpuReservoirContext,
    prepared: &DrainageBudget<'_>,
    upper: &CpuTensor,
) -> Result<CpuTensor, ReservoirError> {
    let c = &context.client;
    let algebraic = c.where_cond(
        &context.instantaneous,
        &c.minimum(prepared.available_depth(), &context.depression)?,
        prepared.available_depth(),
    )?;
    Ok(c.where_cond(
        prepared.active(),
        &c.add(&context.depression, upper)?,
        &algebraic,
    )?)
}
