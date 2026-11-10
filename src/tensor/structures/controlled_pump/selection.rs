//! Compact admitted actual pump operations selected directly from accepted sampled latches.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
#[derive(Clone, Debug)]
pub(crate) struct TensorControlledPumpSelection<R: Runtime<DType = DType>> {
    pub running: Tensor<R>,
    pub speed: Tensor<R>,
    pub power_of_two: Tensor<R>,
    pub high: Tensor<R>,
}
