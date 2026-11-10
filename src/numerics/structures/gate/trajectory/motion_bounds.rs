//! Shape-neutral outward motion arithmetic without global rejection of optional certificates.

use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, ScalarOps, TypeConversionOps, UnaryOps};

pub(super) fn finite(
    context: &CpuStructureContext,
    value: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    let c = &context.client;
    let invalid = c.maximum(
        &c.cast(&c.isnan(value)?, value.dtype())?,
        &c.cast(&c.isinf(value)?, value.dtype())?,
    )?;
    Ok(c.add_scalar(&c.neg(&invalid)?, 1.0)?)
}
pub(super) fn up(
    context: &CpuStructureContext,
    value: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    let bound =
        crate::numerics::subcatchment::bounds::rounding_with_client(&context.client, value)?;
    Ok(context.client.add(value, &bound)?)
}
pub(super) fn down(
    context: &CpuStructureContext,
    value: &CpuTensor,
) -> Result<CpuTensor, StructureError> {
    let bound =
        crate::numerics::subcatchment::bounds::rounding_with_client(&context.client, value)?;
    Ok(context.client.sub(value, &bound)?)
}
