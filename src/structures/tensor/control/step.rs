//! Bounded immediate before/after tensor witnesses without prior-step ownership.

use super::{TensorControlState, TensorControlTimer};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::structures::ControlSample;

#[derive(Debug)]
pub(crate) struct TensorControlStep<R: Runtime<DType = DType>> {
    pub start: f64,
    pub end: f64,
    pub decision_index: Option<usize>,
    pub before: TensorControlState<R>,
    pub after: TensorControlState<R>,
    pub samples: Vec<ControlSample>,
    pub qualifies: Tensor<R>,
    pub timer: TensorControlTimer<R>,
    pub transitioned: Tensor<R>,
}
