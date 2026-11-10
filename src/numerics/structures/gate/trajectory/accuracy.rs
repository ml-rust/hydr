//! Hydraulic opening support and explicit absolute-or-relative receipt admission.

use crate::cpu::{
    CpuTensor,
    structures::{
        CpuStructureContext,
        gate::trajectory::{CpuTrajectoryContext, CpuTrajectoryInput},
    },
};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};

pub(super) fn check(
    context: &CpuStructureContext,
    trajectory: &CpuTrajectoryContext,
    input: &CpuTrajectoryInput,
    opening: &CpuTensor,
    error: &CpuTensor,
) -> Result<(), StructureError> {
    let c = &context.client;
    let gate = context
        .gate
        .as_ref()
        .ok_or_else(|| StructureError::numerical("bound trajectory context lacks gate tensors"))?;
    context.finite(opening, "trajectory opening")?;
    context.finite(error, "trajectory absolute opening error receipt")?;
    context.reject(
        &c.lt(error, &context.zero)?,
        "trajectory error receipt is negative",
    )?;
    let outside = c.maximum(
        &c.lt(opening, &input.minimum)?,
        &c.maximum(
            &c.gt(opening, &input.maximum)?,
            &c.gt(opening, &trajectory.maximum)?,
        )?,
    )?;
    context.reject(
        &c.mul(&gate.mask, &outside)?,
        "trajectory opening exceeds original segment or gate range",
    )?;
    let positive = c.sub(&gate.mask, &input.closed)?;
    context.reject(
        &c.mul(&positive, &c.le(opening, &context.zero)?)?,
        "original positive trajectory support returns zero opening",
    )?;
    context.reject(
        &c.mul(
            &input.closed,
            &c.maximum(&c.ne(opening, &context.zero)?, &c.ne(error, &context.zero)?)?,
        )?,
        "declared closed trajectory support lacks exact zero value and receipt",
    )?;
    let absolute = c.le(error, &trajectory.absolute_limit)?;
    let need_relative = c.mul(&positive, &c.sub(&context.one, &absolute)?)?;
    let relative_active = c.mul(
        &need_relative,
        &c.gt(&trajectory.relative_limit, &context.zero)?,
    )?;
    let numerator = c.where_cond(&relative_active, error, &context.zero)?;
    let denominator = c.where_cond(&relative_active, opening, &context.one)?;
    let relative = crate::numerics::structures::bounds::divide(
        context,
        &numerator,
        &denominator,
        "trajectory relative opening error",
    )?;
    let relative_admitted = c.mul(
        &relative_active,
        &c.le(&relative, &trajectory.relative_limit)?,
    )?;
    context.reject(
        &c.mul(
            &gate.mask,
            &c.sub(&context.one, &c.maximum(&absolute, &relative_admitted)?)?,
        )?,
        "trajectory opening error exceeds both explicit accuracy limits",
    )
}
