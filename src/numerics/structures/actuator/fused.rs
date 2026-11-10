//! Strict fused stages with local arithmetic exactness certificates.

use super::arithmetic::{bound, finite, not, used};
use crate::cpu::{
    CpuTensor,
    structures::actuator::{CpuActuatorContext, CpuActuatorFma},
};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, TwoSumOps, UnaryOps};

pub(super) fn stage(
    ctx: &CpuActuatorContext,
    rate: &CpuTensor,
    time: &CpuTensor,
    addend: &CpuTensor,
    power: &CpuTensor,
    active: &CpuTensor,
) -> Result<CpuActuatorFma, StructureError> {
    let c = &ctx.structures.client;
    let k = c.where_cond(active, rate, &ctx.zero)?;
    let t = c.where_cond(active, time, &ctx.zero)?;
    let x = c.where_cond(active, addend, &ctx.zero)?;
    let result = c.fused_mul_add(&k, &t, &x)?;
    used(
        ctx,
        active,
        &finite(ctx, &result)?,
        "actuator used fused result is nonfinite: rescale inputs or shorten the supported interval",
    )?;
    let threshold = ctx.constant(2.0f64.powi(-485))?;
    let grid = c.mul(
        active,
        &c.mul(
            &c.eq(&result, &ctx.zero)?,
            &c.mul(
                &c.ge(&c.abs(&k)?, &threshold)?,
                &c.ge(&c.abs(&t)?, &threshold)?,
            )?,
        )?,
    )?;
    let candidate = c.mul(active, power)?;
    let product = c.mul(
        &c.where_cond(&candidate, &k, &ctx.zero)?,
        &c.where_cond(&candidate, &t, &ctx.zero)?,
    )?;
    let product_finite = finite(ctx, &product)?;
    let usable = c.mul(&candidate, &product_finite)?;
    let magnitude = c.abs(&k)?;
    let direct = c.maximum(
        &c.ge(&magnitude, &ctx.one)?,
        &c.ge(&c.abs(&product)?, &ctx.constant(f64::MIN_POSITIVE)?)?,
    )?;
    let back_candidate = c.mul(&usable, &not(ctx, &direct)?)?;
    let back = c.div(
        &c.where_cond(&back_candidate, &product, &ctx.zero)?,
        &c.where_cond(&back_candidate, &k, &ctx.one)?,
    )?;
    let exact = c.mul(
        &usable,
        &c.maximum(
            &direct,
            &c.mul(
                &back_candidate,
                &c.mul(&finite(ctx, &back)?, &c.eq(&back, &t)?)?,
            )?,
        )?,
    )?;
    let ph = c.where_cond(&exact, &product, &ctx.zero)?;
    let px = c.where_cond(&exact, &x, &ctx.zero)?;
    let ordinary = c.add(&ph, &px)?;
    let high_finite = finite(ctx, &ordinary)?;
    let witness_candidate = c.mul(&exact, &high_finite)?;
    let (high, low) = c.two_sum(
        &c.where_cond(&witness_candidate, &ph, &ctx.zero)?,
        &c.where_cond(&witness_candidate, &px, &ctx.zero)?,
    )?;
    let witness = c.mul(
        &witness_candidate,
        &c.mul(&finite(ctx, &low)?, &c.eq(&high, &result)?)?,
    )?;
    let zero_product = c.maximum(&c.eq(&k, &ctx.zero)?, &c.eq(&t, &ctx.zero)?)?;
    let exact_zero = c.maximum(&grid, &c.mul(active, &zero_product)?)?;
    let error = c.where_cond(
        &exact_zero,
        &ctx.zero,
        &c.where_cond(&witness, &c.abs(&low)?, &bound(ctx, &result)?)?,
    )?;
    Ok(CpuActuatorFma {
        valid: active.clone(),
        rate: k,
        time: t,
        addend: x,
        result,
        error: c.where_cond(active, &error, &ctx.zero)?,
        zero_grid: grid,
        binary_valid: witness.clone(),
        binary_product: c.where_cond(&witness, &product, &ctx.zero)?,
        binary_high: c.where_cond(&witness, &high, &ctx.zero)?,
        binary_low: c.where_cond(&witness, &low, &ctx.zero)?,
    })
}
