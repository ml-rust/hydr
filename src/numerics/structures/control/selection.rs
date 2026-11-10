//! Tensor-only bounded last-transition selection.

use crate::cpu::{
    CpuTensor,
    structures::control::{CpuControlContext, CpuControlSample, CpuControlTimer},
};
use crate::structures::StructureError;
use numr::ops::ConditionalOps;

pub(super) fn sample(
    context: &CpuControlContext,
    mask: &CpuTensor,
    new: &CpuControlSample,
    old: &CpuControlSample,
) -> Result<CpuControlSample, StructureError> {
    let c = &context.structures.client;
    Ok(CpuControlSample {
        valid: c.where_cond(mask, &new.valid, &old.valid)?,
        observed: c.where_cond(mask, &new.observed, &old.observed)?,
        available: c.where_cond(mask, &new.available, &old.available)?,
        stage: c.where_cond(mask, &new.stage, &old.stage)?,
    })
}
pub(super) fn timer(
    context: &CpuControlContext,
    mask: &CpuTensor,
    new: &CpuControlTimer,
    old: &CpuControlTimer,
) -> Result<CpuControlTimer, StructureError> {
    let c = &context.structures.client;
    macro_rules! choose {
        ($field:ident) => {
            c.where_cond(mask, &new.$field, &old.$field)?
        };
    }
    Ok(CpuControlTimer {
        active: choose!(active),
        started: choose!(started),
        evaluated: choose!(evaluated),
        delay: choose!(delay),
        elapsed_high: choose!(elapsed_high),
        elapsed_low: choose!(elapsed_low),
        ready: choose!(ready),
        allowed_high: choose!(allowed_high),
        allowed_low: choose!(allowed_low),
        overflow: choose!(overflow),
        within: choose!(within),
    })
}
