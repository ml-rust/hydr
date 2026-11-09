//! Shared root witness and conservative hydraulic export receipts.

use crate::cpu::CpuTensor;

#[derive(Debug, Clone)]
pub(crate) struct DrainageRoot {
    pub lower: CpuTensor,
    pub upper: CpuTensor,
    pub depth: CpuTensor,
    pub iterations: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct DrainageResult {
    pub lower: CpuTensor,
    pub upper: CpuTensor,
    pub depth: CpuTensor,
    pub export_depth: CpuTensor,
    pub storage_after: CpuTensor,
    pub export_volume: CpuTensor,
    pub endpoint_rate: CpuTensor,
    pub endpoint_discharge: CpuTensor,
    pub law_depth: CpuTensor,
    pub law_volume: CpuTensor,
    pub constitutive_residual: CpuTensor,
    pub constitutive_scale: CpuTensor,
    pub iterations: usize,
}
