//! Immutable gate geometry, physical-expression error bounds, and flow operands.
use numr::{dtype::DType, runtime::Runtime};

use numr::tensor::Tensor;

#[derive(Clone, Debug)]
pub(crate) struct TensorGateGeometry<R: Runtime<DType = DType>> {
    pub opening: Tensor<R>,
    pub crown_high: Tensor<R>,
    pub crown_low: Tensor<R>,
    pub crown_overflow: Tensor<R>,
    pub submerged_height: Tensor<R>,
    pub submerged_height_bound: Tensor<R>,
    pub free_height: Tensor<R>,
    pub free_height_bound: Tensor<R>,
    pub gap: Tensor<R>,
    pub gap_bound: Tensor<R>,
    pub lower: Tensor<R>,
    pub lower_bound: Tensor<R>,
    pub upper: Tensor<R>,
    pub upper_bound: Tensor<R>,
    pub closure_residual: Tensor<R>,
    pub closure_uncertainty: Tensor<R>,
    pub closure_allowance: Tensor<R>,
}
#[derive(Clone, Debug)]
pub(crate) struct TensorGateResult<R: Runtime<DType = DType>> {
    pub geometry: TensorGateGeometry<R>,
    pub ratio: Tensor<R>,
    pub k: Tensor<R>,
    pub logarithm: Tensor<R>,
}
