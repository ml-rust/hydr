//! Complete persistent arithmetic and stage witnesses for one candidate.

use crate::cpu::CpuTensor;

#[derive(Clone, Debug)]
pub(crate) struct CpuPoolStep {
    pub source: Option<super::CpuPoolSourceStep>,
    pub old_volume_m3: CpuTensor,
    pub incoming_volume_m3: CpuTensor,
    pub outgoing_volume_m3: CpuTensor,
    pub remaining_volume_m3: CpuTensor,
    pub volume_m3: CpuTensor,
    pub outgoing_witness_m3: CpuTensor,
    pub incoming_witness_m3: CpuTensor,
    pub water_residual_m3: CpuTensor,
    pub witness_consistency_m3: CpuTensor,
    pub witness_consistency_bound_m3: CpuTensor,
    pub physical_scale_m3: CpuTensor,
    pub stage_m: CpuTensor,
    pub reconstructed_volume_m3: CpuTensor,
    pub inverse_residual_m3: CpuTensor,
    pub cumulative_incoming_before_m3: CpuTensor,
    pub cumulative_incoming_after_m3: CpuTensor,
    pub cumulative_incoming_witness_m3: CpuTensor,
    pub cumulative_outgoing_before_m3: CpuTensor,
    pub cumulative_outgoing_after_m3: CpuTensor,
    pub cumulative_outgoing_witness_m3: CpuTensor,
    pub incoming_bound_before_m3: CpuTensor,
    pub incoming_bound_after_m3: CpuTensor,
    pub outgoing_bound_before_m3: CpuTensor,
    pub outgoing_bound_after_m3: CpuTensor,
}
