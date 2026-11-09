//! Shared outward arithmetic for independent structure-expression witnesses.

use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};

pub(super) fn rounding(
    context: &CpuStructureContext,
    value: &CpuTensor,
    quantity: &str,
) -> Result<CpuTensor, StructureError> {
    let bound =
        crate::numerics::subcatchment::bounds::rounding_with_client(&context.client, value)?;
    context.finite(&bound, quantity)?;
    Ok(bound)
}

pub(super) fn add(
    context: &CpuStructureContext,
    a: &CpuTensor,
    b: &CpuTensor,
    quantity: &str,
) -> Result<CpuTensor, StructureError> {
    let c = &context.client;
    let a_zero = c.eq(a, &context.zero)?;
    let b_zero = c.eq(b, &context.zero)?;
    let both = c.mul(&c.gt(a, &context.zero)?, &c.gt(b, &context.zero)?)?;
    let sum = c.add(a, b)?;
    context.finite(&sum, quantity)?;
    let used = c.where_cond(&both, &sum, &context.zero)?;
    let padded = c.add(&used, &rounding(context, &used, quantity)?)?;
    let bound = c.where_cond(&a_zero, b, &c.where_cond(&b_zero, a, &padded)?)?;
    context.finite(&bound, quantity)?;
    Ok(bound)
}

pub(super) fn divide(
    context: &CpuStructureContext,
    numerator: &CpuTensor,
    denominator: &CpuTensor,
    quantity: &str,
) -> Result<CpuTensor, StructureError> {
    let c = &context.client;
    let quotient = c.div(numerator, denominator)?;
    context.finite(&quotient, quantity)?;
    let padded = c.add(&quotient, &rounding(context, &quotient, quantity)?)?;
    let bound = c.where_cond(&c.eq(numerator, &context.zero)?, &context.zero, &padded)?;
    context.finite(&bound, quantity)?;
    Ok(bound)
}
