use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Debug)]
pub(crate) struct TensorBoundaryRequest<R: Runtime<DType = DType>> {
    pub stage_depth: Tensor<R>,
    pub outward_discharge: Tensor<R>,
    pub discharge_mask: Tensor<R>,
    pub inflow_tangent: Tensor<R>,
    pub has_inflow_tangent: Tensor<R>,
}

#[derive(Debug, Clone)]
pub(crate) struct TensorBoundaryTrace<R: Runtime<DType = DType>> {
    pub h: Tensor<R>,
    pub q_normal: Tensor<R>,
    /// Primitive velocity remains available when the discharge product underflows.
    pub tangential_velocity: Tensor<R>,
    /// Outward mass, outward-normal momentum, and positive-coordinate tangent momentum.
    pub flux: [Tensor<R>; 3],
    pub speed: Tensor<R>,
    pub discharge_residual: Tensor<R>,
    pub characteristic_residual: Tensor<R>,
}
