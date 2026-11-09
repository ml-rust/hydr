//! Immutable gate geometry, physical-expression error bounds, and flow operands.

use crate::cpu::CpuTensor;

#[derive(Debug)]
pub(crate) struct CpuGateGeometry {
    pub opening: CpuTensor,
    pub crown_high: CpuTensor,
    pub crown_low: CpuTensor,
    pub crown_overflow: CpuTensor,
    pub submerged_height: CpuTensor,
    pub submerged_height_bound: CpuTensor,
    pub free_height: CpuTensor,
    pub free_height_bound: CpuTensor,
    pub gap: CpuTensor,
    pub gap_bound: CpuTensor,
    pub lower: CpuTensor,
    pub lower_bound: CpuTensor,
    pub upper: CpuTensor,
    pub upper_bound: CpuTensor,
    pub closure_residual: CpuTensor,
    pub closure_uncertainty: CpuTensor,
    pub closure_allowance: CpuTensor,
}
#[derive(Debug)]
pub(crate) struct CpuGateResult {
    pub geometry: CpuGateGeometry,
    pub ratio: CpuTensor,
    pub k: CpuTensor,
    pub logarithm: CpuTensor,
}
