//! Positive pressure-strip logarithms combined through Numr logsumexp.

use crate::cpu::structures::{
    CpuStructureContext, CpuStructureInput, CpuStructureResult, gate::CpuGateResult,
};
use crate::structures::StructureError;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, ScalarOps, ShapeOps, UnaryOps,
};

pub(crate) fn evaluate(
    context: &CpuStructureContext,
    input: &CpuStructureInput,
    mut result: CpuStructureResult,
) -> Result<CpuStructureResult, StructureError> {
    let Some(gate) = &context.gate else {
        return Ok(result);
    };
    let operation = input.gate.as_ref().ok_or_else(|| {
        StructureError::numerical("checked gate proposal lacks its explicit opening tensors")
    })?;
    let class = super::classification::classify(context, gate, input, operation)?;
    let geometry = super::geometry::evaluate(context, gate, operation, &class)?;
    let c = &context.client;
    let submerged_height =
        c.where_cond(&class.submerged, &geometry.submerged_height, &context.one)?;
    let gap = c.where_cond(&class.submerged, &geometry.gap, &context.one)?;
    let submerged_log = c.add(
        &c.log(&submerged_height)?,
        &c.mul_scalar(&c.log(&gap)?, 0.5)?,
    )?;
    let free_height = c.where_cond(&class.free, &geometry.free_height, &context.one)?;
    let lower = c.where_cond(&class.free, &geometry.lower, &context.one)?;
    let upper = c.where_cond(&class.free, &geometry.upper, &context.zero)?;
    let ratio = c.div(&upper, &lower)?;
    context.finite(&ratio, "active gate free pressure-head ratio is nonfinite")?;
    context.reject(
        &c.maximum(&c.lt(&ratio, &context.zero)?, &c.gt(&ratio, &context.one)?)?,
        "gate free pressure-head ratio lies outside zero to one",
    )?;
    let k = c.add_scalar(&c.div(&ratio, &c.add_scalar(&c.sqrt(&ratio)?, 1.0)?)?, 1.0)?;
    let free_log = c.add(
        &c.add(
            &c.log(&c.add_scalar(&context.zero, 2.0 / 3.0)?)?,
            &c.log(&free_height)?,
        )?,
        &c.add(&c.mul_scalar(&c.log(&lower)?, 0.5)?, &c.log(&k)?)?,
    )?;
    let absent = c.add_scalar(&context.zero, f64::NEG_INFINITY)?;
    let strips = c.stack(
        &[
            &c.where_cond(&class.submerged, &submerged_log, &absent)?,
            &c.where_cond(&class.free, &free_log, &absent)?,
        ],
        1,
    )?;
    let integral_log = c.logsumexp(&strips, &[1], false)?;
    let raw = c.add(
        &c.add(&gate.log_scale, &gate.log_root_gravity)?,
        &integral_log,
    )?;
    let (magnitude, logarithm) = super::super::orifice::exp_active(context, &class.active, &raw)?;
    let discharge = c.mul(&class.direction, &magnitude)?;
    context.finite(&discharge, "final signed gate proposal is nonfinite")?;
    result.discharge = c.where_cond(&gate.mask, &discharge, &result.discharge)?;
    result.status = c.where_cond(&gate.mask, &class.status, &result.status)?;
    result.regime = c.where_cond(&gate.mask, &class.regime, &result.regime)?;
    result.direction = c.where_cond(&gate.mask, &class.direction, &result.direction)?;
    result.gate = Some(CpuGateResult {
        geometry,
        ratio: c.where_cond(&class.free, &ratio, &context.zero)?,
        k: c.where_cond(&class.free, &k, &context.zero)?,
        logarithm,
    });
    Ok(result)
}
