//! Persistent tensor receipts for one independently admitted reservoir interval.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Debug, Clone)]
pub(crate) struct TensorReservoirStep<R: Runtime<DType = DType>> {
    pub old_depth: Tensor<R>,
    pub incoming_rate: Tensor<R>,
    pub incoming_depth: Tensor<R>,
    pub available_depth: Tensor<R>,
    pub active_duration: Tensor<R>,
    pub root_lower: Tensor<R>,
    pub root_upper: Tensor<R>,
    pub depth: Tensor<R>,
    pub export_depth: Tensor<R>,
    pub storage_before: Tensor<R>,
    pub storage_after: Tensor<R>,
    pub incoming_volume: Tensor<R>,
    pub export_volume: Tensor<R>,
    pub endpoint_rate: Tensor<R>,
    pub endpoint_discharge: Tensor<R>,
    pub law_depth: Tensor<R>,
    pub law_volume: Tensor<R>,
    pub constitutive_residual: Tensor<R>,
    pub constitutive_scale: Tensor<R>,
    pub water_residual: Tensor<R>,
    pub water_scale: Tensor<R>,
    pub root_iterations: usize,
}
