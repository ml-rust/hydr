//! Bounded tensor timing witnesses without nonfinite threshold DTO values.

use super::TensorControlContext;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone, Debug)]
pub(crate) struct TensorControlTimer<R: Runtime<DType = DType>> {
    pub active: Tensor<R>,
    pub started: Tensor<R>,
    pub evaluated: Tensor<R>,
    pub delay: Tensor<R>,
    pub elapsed_high: Tensor<R>,
    pub elapsed_low: Tensor<R>,
    pub ready: Tensor<R>,
    pub allowed_high: Tensor<R>,
    pub allowed_low: Tensor<R>,
    pub overflow: Tensor<R>,
    pub within: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorControlTimer<R> {
    pub(crate) fn zero(context: &TensorControlContext<R>) -> Self {
        Self {
            active: context.zero.clone(),
            started: context.zero.clone(),
            evaluated: context.zero.clone(),
            delay: context.zero.clone(),
            elapsed_high: context.zero.clone(),
            elapsed_low: context.zero.clone(),
            ready: context.zero.clone(),
            allowed_high: context.zero.clone(),
            allowed_low: context.zero.clone(),
            overflow: context.zero.clone(),
            within: context.zero.clone(),
        }
    }
}
