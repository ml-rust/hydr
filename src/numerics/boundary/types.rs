use crate::cpu::CpuTensor;

#[derive(Debug)]
pub(crate) struct TensorBoundaryRequest {
    pub stage_depth: CpuTensor,
    pub outward_discharge: CpuTensor,
    pub discharge_mask: CpuTensor,
    pub inflow_tangent: CpuTensor,
    pub has_inflow_tangent: CpuTensor,
}

#[derive(Debug)]
pub(crate) struct TensorBoundaryTrace {
    pub h: CpuTensor,
    pub q_normal: CpuTensor,
    /// Primitive velocity remains available when the discharge product underflows.
    pub tangential_velocity: CpuTensor,
    /// Outward mass, outward-normal momentum, and positive-coordinate tangent momentum.
    pub flux: [CpuTensor; 3],
    pub speed: CpuTensor,
    pub discharge_residual: CpuTensor,
    pub characteristic_residual: CpuTensor,
}
