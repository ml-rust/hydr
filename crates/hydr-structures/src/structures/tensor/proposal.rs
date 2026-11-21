//! Persistent proposal tensors with no evolving storage or accepted-transfer state.
use numr::{dtype::DType, runtime::Runtime};

use numr::tensor::Tensor;

#[doc(hidden)]
#[derive(Debug)]
pub struct TensorStructureResult<R: Runtime<DType = DType>> {
    pub(crate) culvert: Option<super::culvert::TensorCulvertResult<R>>,
    pub(crate) gate: Option<super::gate::TensorGateResult<R>>,
    pub(crate) pump: Option<super::pump::TensorPumpResult<R>>,
    pub discharge: Tensor<R>,
    pub regime: Tensor<R>,
    pub status: Tensor<R>,
    pub direction: Tensor<R>,
    pub head_gap: Tensor<R>,
    pub upstream_head: Tensor<R>,
    pub downstream_head: Tensor<R>,
    pub ratio: Tensor<R>,
    pub correction_k: Tensor<R>,
    pub log_magnitude: Tensor<R>,
    /// Bound on `|dQ/dH| / |Q|` per metre, for a move of either endpoint head.
    ///
    /// Zero where the element carries no discharge.
    pub head_sensitivity: Tensor<R>,
}
