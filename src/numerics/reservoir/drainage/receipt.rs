//! Shared root witness and conservative hydraulic export receipts.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Debug, Clone)]
pub(crate) struct DrainageRoot<R: Runtime<DType = DType>> {
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
    pub depth: Tensor<R>,
    pub iterations: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct DrainageResult<R: Runtime<DType = DType>> {
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
    pub depth: Tensor<R>,
    pub export_depth: Tensor<R>,
    pub storage_after: Tensor<R>,
    pub export_volume: Tensor<R>,
    pub endpoint_rate: Tensor<R>,
    pub endpoint_discharge: Tensor<R>,
    pub law_depth: Tensor<R>,
    pub law_volume: Tensor<R>,
    pub constitutive_residual: Tensor<R>,
    pub constitutive_scale: Tensor<R>,
    pub iterations: usize,
}
