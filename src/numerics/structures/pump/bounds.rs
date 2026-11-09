//! Source-specific outward head-normalization bounds with exact zero identities.

use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};

pub(super) fn rounding(
    context: &CpuStructureContext,
    value: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    let bound =
        crate::numerics::subcatchment::bounds::rounding_with_client(&context.client, value)?;
    context.finite(&bound, "pump normalization uncertainty bound overflows")?;
    Ok(bound)
}

pub(super) fn add(
    context: &CpuStructureContext,
    a: &CpuTensor,
    b: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    let c = &context.client;
    let a_zero = c.eq(a, &context.zero)?;
    let b_zero = c.eq(b, &context.zero)?;
    let both = c.mul(&c.gt(a, &context.zero)?, &c.gt(b, &context.zero)?)?;
    let sum = c.add(a, b)?;
    context.finite(&sum, "pump normalization uncertainty bound overflows")?;
    let used = c.where_cond(&both, &sum, &context.zero)?;
    let padded = c.add(&used, &rounding(context, &used)?)?;
    let bound = c.where_cond(&a_zero, b, &c.where_cond(&b_zero, a, &padded)?)?;
    context.finite(&bound, "pump normalization uncertainty bound overflows")?;
    Ok(bound)
}

pub(super) fn divide(
    context: &CpuStructureContext,
    numerator: &CpuTensor,
    denominator: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    let c = &context.client;
    let quotient = c.div(numerator, denominator)?;
    context.finite(&quotient, "pump normalization uncertainty bound overflows")?;
    let padded = c.add(&quotient, &rounding(context, &quotient)?)?;
    let bound = c.where_cond(&c.eq(numerator, &context.zero)?, &context.zero, &padded)?;
    context.finite(&bound, "pump normalization uncertainty bound overflows")?;
    Ok(bound)
}

pub(super) fn interval(
    context: &CpuStructureContext,
    query: &CpuTensor,
    bound: &CpuTensor,
) -> Result<(CpuTensor, CpuTensor), StructureError> {
    let c = &context.client;
    let has_bound = c.gt(bound, &context.zero)?;
    let center = c.where_cond(&has_bound, query, &context.zero)?;
    let lo0 = c.sub(&center, bound)?;
    let hi0 = c.add(&center, bound)?;
    context.finite(&lo0, "pump normalization lower interval overflows")?;
    context.finite(&hi0, "pump normalization upper interval overflows")?;
    let lower = c.sub(&lo0, &rounding(context, &lo0)?)?;
    let upper = c.add(&hi0, &rounding(context, &hi0)?)?;
    context.finite(&lower, "pump normalization lower interval overflows")?;
    context.finite(&upper, "pump normalization upper interval overflows")?;
    Ok((
        c.where_cond(&has_bound, &lower, query)?,
        c.where_cond(&has_bound, &upper, query)?,
    ))
}
