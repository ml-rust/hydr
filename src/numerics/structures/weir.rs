//! Stable all-log suppressed rectangular weir laws with direct original-stage gap.

use super::direction::Classification;
use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, UnaryOps};

const VILLEMONTE_EXPONENT: f64 = 0.385;

pub(super) struct WeirResult {
    pub magnitude: CpuTensor,
    pub logarithm: CpuTensor,
    pub upstream: CpuTensor,
    pub downstream: CpuTensor,
    pub gap: CpuTensor,
    pub ratio: CpuTensor,
    pub k: CpuTensor,
}

pub(super) fn free(
    context: &CpuStructureContext,
    class: &Classification,
) -> Result<WeirResult, StructureError> {
    let c = &context.client;
    let active = &class.free_weir;
    let high = c.where_cond(active, &class.high, &context.one)?;
    let crest = c.where_cond(active, &context.crest, &context.zero)?;
    let hu = c.sub(&high, &crest)?;
    context.positive(
        &hu,
        "active free-overflow head above crest must remain positive finite",
    )?;
    let raw = c.add(&context.log_weir_scale, &c.mul_scalar(&c.log(&hu)?, 1.5)?)?;
    let (magnitude, logarithm) = super::orifice::exp_active(context, active, &raw)?;
    Ok(WeirResult {
        magnitude,
        logarithm,
        upstream: c.where_cond(active, &hu, &context.zero)?,
        downstream: context.zero.clone(),
        gap: context.zero.clone(),
        ratio: context.zero.clone(),
        k: context.zero.clone(),
    })
}

pub(super) fn submerged(
    context: &CpuStructureContext,
    class: &Classification,
) -> Result<WeirResult, StructureError> {
    let c = &context.client;
    let active = &class.submerged_weir;
    let high = c.where_cond(active, &class.high, &context.one)?;
    let low = c.where_cond(active, &class.low, &c.mul_scalar(&context.one, 0.5)?)?;
    let crest = c.where_cond(active, &context.crest, &context.zero)?;
    let hu = c.sub(&high, &crest)?;
    let hd = c.sub(&low, &crest)?;
    let gap = c.sub(&high, &low)?;
    context.positive(
        &hu,
        "active submerged upstream head must remain positive finite",
    )?;
    context.positive(
        &hd,
        "active submerged downstream head must remain positive finite",
    )?;
    context.positive(
        &gap,
        "active direct original-stage gap must remain positive finite",
    )?;
    let ratio = c.div(&hd, &hu)?;
    context.finite(&ratio, "represented submerged head ratio")?;
    context.reject(
        &c.maximum(&c.lt(&ratio, &context.zero)?, &c.gt(&ratio, &context.one)?)?,
        "represented submerged head ratio lies outside [0,1]",
    )?;
    let k = c.add(
        &context.one,
        &c.div(&ratio, &c.add(&context.one, &c.sqrt(&ratio)?)?)?,
    )?;
    let raw = c.add(
        &c.add(
            &c.add(
                &context.log_weir_scale,
                &c.mul_scalar(&c.log(&hu)?, 1.5 - VILLEMONTE_EXPONENT)?,
            )?,
            &c.mul_scalar(&c.log(&gap)?, VILLEMONTE_EXPONENT)?,
        )?,
        &c.mul_scalar(&c.log(&k)?, VILLEMONTE_EXPONENT)?,
    )?;
    let (magnitude, logarithm) = super::orifice::exp_active(context, active, &raw)?;
    Ok(WeirResult {
        magnitude,
        logarithm,
        upstream: c.where_cond(active, &hu, &context.zero)?,
        downstream: c.where_cond(active, &hd, &context.zero)?,
        gap: c.where_cond(active, &gap, &context.zero)?,
        ratio: c.where_cond(active, &ratio, &context.zero)?,
        k: c.where_cond(active, &k, &context.zero)?,
    })
}
