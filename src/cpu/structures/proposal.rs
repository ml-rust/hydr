//! Persistent proposal tensors with no evolving storage or accepted-transfer state.

use crate::cpu::CpuTensor;

#[derive(Debug)]
pub(crate) struct CpuStructureResult {
    pub gate: Option<super::gate::CpuGateResult>,
    pub pump: Option<super::pump::CpuPumpResult>,
    pub discharge: CpuTensor,
    pub regime: CpuTensor,
    pub status: CpuTensor,
    pub direction: CpuTensor,
    pub head_gap: CpuTensor,
    pub upstream_head: CpuTensor,
    pub downstream_head: CpuTensor,
    pub ratio: CpuTensor,
    pub correction_k: CpuTensor,
    pub log_magnitude: CpuTensor,
}
