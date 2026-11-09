//! Persistent tensor receipts for one independently admitted reservoir interval.

use crate::cpu::CpuTensor;

#[derive(Debug, Clone)]
pub(crate) struct CpuReservoirStep {
    pub old_depth: CpuTensor,
    pub incoming_rate: CpuTensor,
    pub incoming_depth: CpuTensor,
    pub available_depth: CpuTensor,
    pub active_duration: CpuTensor,
    pub root_lower: CpuTensor,
    pub root_upper: CpuTensor,
    pub depth: CpuTensor,
    pub export_depth: CpuTensor,
    pub storage_before: CpuTensor,
    pub storage_after: CpuTensor,
    pub incoming_volume: CpuTensor,
    pub export_volume: CpuTensor,
    pub endpoint_rate: CpuTensor,
    pub endpoint_discharge: CpuTensor,
    pub law_depth: CpuTensor,
    pub law_volume: CpuTensor,
    pub constitutive_residual: CpuTensor,
    pub constitutive_scale: CpuTensor,
    pub water_residual: CpuTensor,
    pub water_scale: CpuTensor,
    pub root_iterations: usize,
}
