//! All-log fully submerged orifice evaluation without unnecessary physical products.

use super::direction::Classification;
use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, ConditionalOps, ScalarOps, UnaryOps};

pub(super) struct OrificeResult {
    pub magnitude: CpuTensor,
    pub logarithm: CpuTensor,
    pub gap: CpuTensor,
}

pub(super) fn exp_active(
    context: &CpuStructureContext,
    active: &CpuTensor,
    raw: &CpuTensor,
) -> Result<(CpuTensor, CpuTensor), StructureError> {
    let c = &context.client;
    let logarithm = c.where_cond(active, raw, &context.zero)?;
    context.finite(&logarithm, "active log discharge")?;
    let magnitude = c.exp(&logarithm)?;
    context.positive(
        &magnitude,
        "active final discharge overflows or underflows to zero",
    )?;
    Ok((c.where_cond(active, &magnitude, &context.zero)?, logarithm))
}

pub(super) fn evaluate(
    context: &CpuStructureContext,
    class: &Classification,
) -> Result<OrificeResult, StructureError> {
    let c = &context.client;
    let high = c.where_cond(&class.orifice, &class.high, &context.one)?;
    let low = c.where_cond(&class.orifice, &class.low, &context.zero)?;
    let gap = c.sub(&high, &low)?;
    context.positive(
        &gap,
        "active original orifice head gap must remain positive finite",
    )?;
    let root_log = c.mul_scalar(
        &c.add(
            &c.add(&context.log_two, &context.log_gravity)?,
            &c.log(&gap)?,
        )?,
        0.5,
    )?;
    let raw = c.add(&context.log_orifice_scale, &root_log)?;
    let (magnitude, logarithm) = exp_active(context, &class.orifice, &raw)?;
    Ok(OrificeResult {
        magnitude,
        logarithm,
        gap: c.where_cond(&class.orifice, &gap, &context.zero)?,
    })
}
