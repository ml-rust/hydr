//! Pinned EPA infiltration integral with solvr correction iteration.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::storage::TensorSoilContext;
use crate::soil::SoilError;
use solvr::optimize::roots::{BatchedNewtonAlgorithms, BatchedNewtonOptions};

/// Conditional MAX preserves the source comparison and tie behavior.
pub(super) fn native_max<R>(
    context: &TensorSoilContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>, SoilError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedNewtonAlgorithms<R>,
{
    Ok(context.client.where_cond(&context.client.gt(a, b)?, a, b)?)
}

/// Conditional MIN preserves the source comparison and tie behavior.
pub(super) fn native_min<R>(
    context: &TensorSoilContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>, SoilError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedNewtonAlgorithms<R>,
{
    Ok(context.client.where_cond(&context.client.lt(a, b)?, a, b)?)
}

pub(super) fn integral<R>(
    context: &TensorSoilContext<R>,
    f1: &Tensor<R>,
    c1: &Tensor<R>,
    ts: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<Tensor<R>, SoilError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedNewtonAlgorithms<R>,
{
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
