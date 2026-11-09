//! Original signed Q-times-duration proposals without donor or storage admission.

use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps, ScalarOps, UnaryOps};

pub(crate) fn volumes(
    context: &CpuStructureContext,
    discharge: &CpuTensor,
    dt: f64,
) -> Result<CpuTensor, StructureError> {
    let c = &context.client;
    let volume = c.mul_scalar(discharge, dt)?;
    context.finite(&volume, "held-head signed Q*dt proposal overflow")?;
    context.reject(
        &c.mul(
            &c.gt(&c.abs(discharge)?, &context.zero)?,
            &c.eq(&volume, &context.zero)?,
        )?,
        "active held-head volume underflows to zero",
    )?;
    Ok(volume)
}
