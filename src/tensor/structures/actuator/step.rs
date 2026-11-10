//! One bounded accepted actuator interval without prior-step ownership.

use super::{
    TensorActuatorForward, TensorActuatorPosition, TensorActuatorState, TensorActuatorUpdate,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::structures::GateActuatorEvent;

#[derive(Clone, Debug)]
pub(crate) struct TensorActuatorStep<R: Runtime<DType = DType>> {
    pub start: f64,
    pub end: f64,
    pub event: Option<GateActuatorEvent>,
    pub before: TensorActuatorState<R>,
    pub after: TensorActuatorState<R>,
    pub left: TensorActuatorPosition<R>,
    pub forward: TensorActuatorForward<R>,
    pub update: TensorActuatorUpdate<R>,
    pub completed: Tensor<R>,
    pub changed: Tensor<R>,
}
