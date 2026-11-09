//! Immutable evaluated pump head-normalization and curve operands.

use crate::cpu::CpuTensor;

#[derive(Debug)]
pub(crate) struct CpuPumpResult {
    pub head_high: CpuTensor,
    pub head_low: CpuTensor,
    pub reference_head: CpuTensor,
    pub error_bound: CpuTensor,
    pub lower: CpuTensor,
    pub upper: CpuTensor,
    pub reference_discharge: CpuTensor,
    pub division_exact: CpuTensor,
}
