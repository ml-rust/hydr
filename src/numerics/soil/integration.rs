//! Pinned EPA infiltration integral with solvr correction iteration.

use crate::cpu::CpuTensor;
use crate::cpu::soil_storage::CpuSoilContext;
use crate::soil::SoilError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, UnaryOps};
use solvr::optimize::roots::{BatchedNewtonAlgorithms, BatchedNewtonOptions};

/// Conditional MAX preserves the source comparison and tie behavior.
pub(super) fn native_max(
    context: &CpuSoilContext,
    a: &CpuTensor,
    b: &CpuTensor,
) -> Result<CpuTensor, SoilError> {
    Ok(context.client.where_cond(&context.client.gt(a, b)?, a, b)?)
}

/// Conditional MIN preserves the source comparison and tie behavior.
pub(super) fn native_min(
    context: &CpuSoilContext,
    a: &CpuTensor,
    b: &CpuTensor,
) -> Result<CpuTensor, SoilError> {
    Ok(context.client.where_cond(&context.client.lt(a, b)?, a, b)?)
}

pub(super) fn integral(
    context: &CpuSoilContext,
    f1: &CpuTensor,
    c1: &CpuTensor,
    ts: &CpuTensor,
    active: &CpuTensor,
) -> Result<CpuTensor, SoilError> {
    let c = &context.client;
    let f1 = c.where_cond(active, f1, &context.one)?;
    let c1 = c.where_cond(active, c1, &context.zero)?;
    let ts = c.where_cond(active, ts, &context.zero)?;
    let ks_ts = c.mul(&context.ks, &ts)?;
    let minimum = c.add(&f1, &ks_ts)?;
    let nonzero = c.ne(&c1, &context.zero)?;
    let short = c.mul(
        &c.lt(&ts, &c.mul_scalar(&context.one, 10.0)?)?,
        &c.gt(&f1, &c.mul_scalar(&c1, 0.01)?)?,
    )?;
    let short = c.mul(&nonzero, &short)?;
    let short_f1 = c.where_cond(&short, &f1, &context.one)?;
    let short_c1 = c.where_cond(&short, &c1, &context.zero)?;
    // Keep native order: f1 + ks * (1 + c1/f1) * ts.
    let direct = c.add(
        &f1,
        &c.mul(
            &c.mul(
                &context.ks,
                &c.add_scalar(&c.div(&short_c1, &short_f1)?, 1.0)?,
            )?,
            &ts,
        )?,
    )?;
    let newton_mask = c.mul(&nonzero, &c.sub(&context.one, &short)?)?;
    let initial = c.where_cond(&newton_mask, &f1, &context.one)?;
    let coefficient = c.where_cond(&newton_mask, &c1, &context.zero)?;
    let duration_rate = c.where_cond(&newton_mask, &ks_ts, &context.zero)?;
    let c2 = c.sub(
        &c.mul(&coefficient, &c.log(&c.add(&initial, &coefficient)?)?)?,
        &duration_rate,
    )?;
    let result = c.newton_batched(
        |x| {
            let sum = c.add(x, &coefficient)?;
            let numerator = c.add(
                &c.sub(&c.sub(x, &initial)?, &c.mul(&coefficient, &c.log(&sum)?)?)?,
                &c2,
            )?;
            let denominator = c.rsub_scalar(&c.div(&coefficient, &sum)?, 1.0)?;
            c.div(&numerator, &denominator)
        },
        &initial,
        &BatchedNewtonOptions {
            correction_tolerance: 1e-5,
            max_iter: 20,
        },
    )?;
    // Invalid active logarithms and singular derivatives select the native F2min fallback.
    let integrated = c.where_cond(
        &result.converged,
        &native_max(context, &result.x, &minimum)?,
        &minimum,
    )?;
    let integrated = c.where_cond(
        &short,
        &native_max(context, &direct, &minimum)?,
        &integrated,
    )?;
    Ok(c.where_cond(&nonzero, &integrated, &minimum)?)
}
