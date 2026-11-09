//! Independent gate-expression uncertainty and resolved-positive operand admission.

use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::StructureError;
use numr::ops::{CompareOps, ConditionalOps, ScalarOps};

const QUANTITY: &str = "gate geometry uncertainty bound overflows";

pub(super) fn add(
    context: &CpuStructureContext,
    a: &CpuTensor,
    b: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    super::super::bounds::add(context, a, b, QUANTITY)
}

pub(super) fn resolved_positive(
    context: &CpuStructureContext,
    mask: &CpuTensor,
    value: &CpuTensor,
    error: &CpuTensor,
) -> Result<(), StructureError> {
    let c = &context.client;
    let center = c.where_cond(mask, value, &context.one)?;
    let bound = c.where_cond(mask, error, &context.zero)?;
    context.positive(
        &center,
        "active positive gate geometry is unresolved or nonfinite",
    )?;
    context.finite(&bound, QUANTITY)?;
    context.reject(
        &c.lt(&bound, &context.zero)?,
        "gate geometry bound must remain nonnegative",
    )?;
    let ratio = super::super::bounds::divide(context, &bound, &center, QUANTITY)?;
    context.reject(
        &c.gt(&ratio, &c.add_scalar(&context.zero, 64.0 * f64::EPSILON)?)?,
        "active gate geometry uncertainty exceeds 64*EPSILON relative admission",
    )
}
