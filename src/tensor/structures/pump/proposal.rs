//! Immutable evaluated pump head-normalization and curve operands.
use numr::{dtype::DType, runtime::Runtime};

use numr::tensor::Tensor;

#[derive(Debug)]
pub(crate) struct TensorPumpResult<R: Runtime<DType = DType>> {
    pub head_high: Tensor<R>,
    pub head_low: Tensor<R>,
    pub reference_head: Tensor<R>,
    pub error_bound: Tensor<R>,
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
    pub reference_discharge: Tensor<R>,
    pub division_exact: Tensor<R>,
}
