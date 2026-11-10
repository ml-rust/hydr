//! Bounded tensor timing witnesses without nonfinite threshold DTO values.

use super::CpuControlContext;
use crate::cpu::CpuTensor;

#[derive(Clone, Debug)]
pub(crate) struct CpuControlTimer {
    pub active: CpuTensor,
    pub started: CpuTensor,
    pub evaluated: CpuTensor,
    pub delay: CpuTensor,
    pub elapsed_high: CpuTensor,
    pub elapsed_low: CpuTensor,
    pub ready: CpuTensor,
    pub allowed_high: CpuTensor,
    pub allowed_low: CpuTensor,
    pub overflow: CpuTensor,
    pub within: CpuTensor,
}
impl CpuControlTimer {
    pub(crate) fn zero(context: &CpuControlContext) -> Self {
        Self {
            active: context.zero.clone(),
            started: context.zero.clone(),
            evaluated: context.zero.clone(),
            delay: context.zero.clone(),
            elapsed_high: context.zero.clone(),
            elapsed_low: context.zero.clone(),
            ready: context.zero.clone(),
            allowed_high: context.zero.clone(),
            allowed_low: context.zero.clone(),
            overflow: context.zero.clone(),
            within: context.zero.clone(),
        }
    }
}
