//! Fixed-orientation running/shutoff/isolation proposals with immutable operating metadata.

use crate::cpu::structures::{
    CpuStructureContext, CpuStructureInput, CpuStructureResult, codes, pump::CpuPumpResult,
};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps};

pub(crate) fn evaluate(
    context: &CpuStructureContext,
    input: &CpuStructureInput,
    mut result: CpuStructureResult,
) -> Result<CpuStructureResult, StructureError> {
    let Some(pump) = &context.pump else {
        return Ok(result);
    };
    let operation = input.pump.as_ref().ok_or_else(|| {
        StructureError::numerical("checked pump proposal lacks its explicit operating tensors")
    })?;
    let c = &context.client;
    let (donor, receiver) = super::admission::heads(context, pump, input, operation)?;
    let normalized = super::normalization::evaluate(context, &donor, &receiver, operation)?;
    let (reference, zero_region) = super::curve::evaluate(context, pump, operation, &normalized)?;
    let positive = c.sub(&operation.running, &zero_region)?;
    let used_speed = c.where_cond(&positive, &operation.speed, &context.one)?;
    let used_reference = c.where_cond(&positive, &reference, &context.zero)?;
    let magnitude = c.mul(&used_speed, &used_reference)?;
    context.finite(
        &magnitude,
        "active final pump discharge overflows or underflows to zero",
    )?;
    context.reject(
        &c.mul(&positive, &c.le(&magnitude, &context.zero)?)?,
        "active final pump discharge overflows or underflows to zero",
    )?;
    let discharge = c.where_cond(
        &positive,
        &c.mul(&pump.orientation, &magnitude)?,
        &context.zero,
    )?;
    let status = c.add(
        &c.mul_scalar(&positive, codes::PUMP_RUNNING)?,
        &c.add(
            &c.mul_scalar(&zero_region, codes::PUMP_SHUTOFF)?,
            &c.mul_scalar(&operation.isolated, codes::PUMP_STOPPED)?,
        )?,
    )?;
    let regime = c.add(
        &c.mul_scalar(&operation.running, codes::PUMP_CURVE)?,
        &c.mul_scalar(&operation.isolated, codes::PUMP_ISOLATED)?,
    )?;
    result.discharge = c.where_cond(&pump.mask, &discharge, &result.discharge)?;
    result.status = c.where_cond(&pump.mask, &status, &result.status)?;
    result.regime = c.where_cond(&pump.mask, &regime, &result.regime)?;
    result.direction = c.where_cond(
        &pump.mask,
        &c.where_cond(&positive, &pump.orientation, &context.zero)?,
        &result.direction,
    )?;
    result.pump = Some(CpuPumpResult {
        head_high: normalized.high,
        head_low: normalized.low,
        reference_head: normalized.query,
        error_bound: normalized.bound,
        lower: normalized.lower,
        upper: normalized.upper,
        reference_discharge: reference,
        division_exact: normalized.division_exact,
    });
    Ok(result)
}
