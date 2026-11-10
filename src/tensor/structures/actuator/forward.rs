//! Persistent batched finite-rate forward tensors.

use super::record::record;
use super::*;
use numr::tensor::Tensor;

record!(
    TensorActuatorFma<R>,
    TensorActuatorFmaValues {
        valid: Tensor<R>,
        rate: Tensor<R>,
        time: Tensor<R>,
        addend: Tensor<R>,
        result: Tensor<R>,
        error: Tensor<R>,
        zero_grid: Tensor<R>,
        binary_valid: Tensor<R>,
        binary_product: Tensor<R>,
        binary_high: Tensor<R>,
        binary_low: Tensor<R>,
    }
);
record!(
    TensorActuatorMotion<R>,
    TensorActuatorMotionValues {
        valid: Tensor<R>,
        direction: Tensor<R>,
        rate: Tensor<R>,
        primary: TensorActuatorFma<R>,
        low: TensorActuatorFma<R>,
        error: Tensor<R>,
        raw_lower_valid: Tensor<R>,
        raw_lower: Tensor<R>,
        raw_upper_valid: Tensor<R>,
        raw_upper: Tensor<R>,
    }
);
record!(
    TensorActuatorEndpoint<R>,
    TensorActuatorEndpointValues {
        anchor: Tensor<R>,
        motion: TensorActuatorMotion<R>,
        lower: Tensor<R>,
        upper: Tensor<R>,
    }
);
record!(
    TensorActuatorForward<R>,
    TensorActuatorForwardValues {
        valid: Tensor<R>,
        anchor: TensorActuatorAnchor<R>,
        query: Tensor<R>,
        target: Tensor<R>,
        elapsed_high: Tensor<R>,
        elapsed_low: Tensor<R>,
        lower_endpoint: TensorActuatorEndpoint<R>,
        upper_endpoint: TensorActuatorEndpoint<R>,
        center_motion: TensorActuatorMotion<R>,
        position: TensorActuatorPosition<R>,
    }
);

pub(crate) struct TensorActuatorFmaValues {
    pub valid: Vec<f64>,
    pub rate: Vec<f64>,
    pub time: Vec<f64>,
    pub addend: Vec<f64>,
    pub result: Vec<f64>,
    pub error: Vec<f64>,
    pub zero_grid: Vec<f64>,
    pub binary_valid: Vec<f64>,
    pub binary_product: Vec<f64>,
    pub binary_high: Vec<f64>,
    pub binary_low: Vec<f64>,
}

pub(crate) struct TensorActuatorMotionValues {
    pub valid: Vec<f64>,
    pub direction: Vec<f64>,
    pub rate: Vec<f64>,
    pub primary: TensorActuatorFmaValues,
    pub low: TensorActuatorFmaValues,
    pub error: Vec<f64>,
    pub raw_lower_valid: Vec<f64>,
    pub raw_lower: Vec<f64>,
    pub raw_upper_valid: Vec<f64>,
    pub raw_upper: Vec<f64>,
}

pub(crate) struct TensorActuatorEndpointValues {
    pub anchor: Vec<f64>,
    pub motion: TensorActuatorMotionValues,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

pub(crate) struct TensorActuatorForwardValues {
    pub valid: Vec<f64>,
    pub anchor: TensorActuatorAnchorValues,
    pub query: Vec<f64>,
    pub target: Vec<f64>,
    pub elapsed_high: Vec<f64>,
    pub elapsed_low: Vec<f64>,
    pub lower_endpoint: TensorActuatorEndpointValues,
    pub upper_endpoint: TensorActuatorEndpointValues,
    pub center_motion: TensorActuatorMotionValues,
    pub position: TensorActuatorPositionValues,
}
