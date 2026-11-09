//! Source-specific outward head-normalization bounds with exact zero identities.

use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};

pub(super) fn rounding(
    context: &CpuStructureContext,
    value: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    super::super::bounds::rounding(
        context,
        value,
        "pump normalization uncertainty bound overflows",
    )
}

pub(super) fn add(
    context: &CpuStructureContext,
    a: &CpuTensor,
    b: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    super::super::bounds::add(
        context,
        a,
        b,
        "pump normalization uncertainty bound overflows",
    )
}

pub(super) fn divide(
    context: &CpuStructureContext,
    numerator: &CpuTensor,
    denominator: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    super::super::bounds::divide(
        context,
        numerator,
        denominator,
        "pump normalization uncertainty bound overflows",
    )
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
