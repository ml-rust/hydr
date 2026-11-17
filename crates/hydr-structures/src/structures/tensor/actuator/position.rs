//! Persistent batched finite-rate position tensors.

use super::record::record;
use numr::tensor::Tensor;

record!(
    TensorActuatorPosition<R>,
    TensorActuatorPositionValues {
        center: Tensor<R>,
        lower: Tensor<R>,
        upper: Tensor<R>,
        error: Tensor<R>,
    }
);
record!(
    TensorActuatorAnchor<R>,
    TensorActuatorAnchorValues {
        epoch: Tensor<R>,
        position: TensorActuatorPosition<R>,
    }
);
record!(
    TensorActuatorArrival<R>,
    TensorActuatorArrivalValues {
        kind: Tensor<R>,
        lower: Tensor<R>,
        upper: Tensor<R>,
        width_high: Tensor<R>,
        width_low: Tensor<R>,
    }
);

pub(crate) struct TensorActuatorPositionValues {
    pub center: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub error: Vec<f64>,
}

pub(crate) struct TensorActuatorAnchorValues {
    pub epoch: Vec<f64>,
    pub position: TensorActuatorPositionValues,
}

pub(crate) struct TensorActuatorArrivalValues {
    pub kind: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
    pub width_high: Vec<f64>,
    pub width_low: Vec<f64>,
}
