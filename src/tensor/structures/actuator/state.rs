//! Persistent batched state with heap-owned immutable bounded receipts.

use super::record::record;
use super::*;
use numr::tensor::Tensor;

use std::sync::Arc;

record!(
    TensorActuatorCompletion<R>,
    TensorActuatorCompletionValues {
        valid: Tensor<R>,
        episode: Tensor<R>,
        generation: Tensor<R>,
        confirmed: Tensor<R>,
        anchor: TensorActuatorAnchor<R>,
        target: Tensor<R>,
        arrival: TensorActuatorArrival<R>,
        effective_upper: Tensor<R>,
        width_high: Tensor<R>,
        width_low: Tensor<R>,
        forward: Arc<TensorActuatorForward<R>>,
    }
);
record!(
    TensorActuatorUpdate<R>,
    TensorActuatorUpdateValues {
        valid: Tensor<R>,
        target: Tensor<R>,
        issued: Tensor<R>,
        available: Tensor<R>,
    }
);
record!(
    TensorActuatorChange<R>,
    TensorActuatorChangeValues {
        valid: Tensor<R>,
        index: Tensor<R>,
        time: Tensor<R>,
        previous_target: Tensor<R>,
        previous_anchor: TensorActuatorAnchor<R>,
        previous_completion: TensorActuatorCompletion<R>,
        update: TensorActuatorUpdate<R>,
        left: TensorActuatorPosition<R>,
        new_anchor: TensorActuatorAnchor<R>,
        generation: Tensor<R>,
        new_episode_valid: Tensor<R>,
        new_episode: Tensor<R>,
    }
);
record!(
    TensorActuatorState<R>,
    TensorActuatorStateValues {
        anchor: TensorActuatorAnchor<R>,
        target: Tensor<R>,
        position: TensorActuatorPosition<R>,
        phase: Tensor<R>,
        arrival: TensorActuatorArrival<R>,
        generation: Tensor<R>,
        episodes: Tensor<R>,
        arrivals: Tensor<R>,
        active_valid: Tensor<R>,
        active_index: Tensor<R>,
        latest_update: TensorActuatorUpdate<R>,
        last_change: Arc<TensorActuatorChange<R>>,
        last_completion: Arc<TensorActuatorCompletion<R>>,
    }
);

pub(crate) struct TensorActuatorCompletionValues {
    pub valid: Vec<f64>,
    pub episode: Vec<f64>,
    pub generation: Vec<f64>,
    pub confirmed: Vec<f64>,
    pub anchor: TensorActuatorAnchorValues,
    pub target: Vec<f64>,
    pub arrival: TensorActuatorArrivalValues,
    pub effective_upper: Vec<f64>,
    pub width_high: Vec<f64>,
    pub width_low: Vec<f64>,
    pub forward: TensorActuatorForwardValues,
}

pub(crate) struct TensorActuatorUpdateValues {
    pub valid: Vec<f64>,
    pub target: Vec<f64>,
    pub issued: Vec<f64>,
    pub available: Vec<f64>,
}

pub(crate) struct TensorActuatorChangeValues {
    pub valid: Vec<f64>,
    pub index: Vec<f64>,
    pub time: Vec<f64>,
    pub previous_target: Vec<f64>,
    pub previous_anchor: TensorActuatorAnchorValues,
    pub previous_completion: TensorActuatorCompletionValues,
    pub update: TensorActuatorUpdateValues,
    pub left: TensorActuatorPositionValues,
    pub new_anchor: TensorActuatorAnchorValues,
    pub generation: Vec<f64>,
    pub new_episode_valid: Vec<f64>,
    pub new_episode: Vec<f64>,
}

pub(crate) struct TensorActuatorStateValues {
    pub anchor: TensorActuatorAnchorValues,
    pub target: Vec<f64>,
    pub position: TensorActuatorPositionValues,
    pub phase: Vec<f64>,
    pub arrival: TensorActuatorArrivalValues,
    pub generation: Vec<f64>,
    pub episodes: Vec<f64>,
    pub arrivals: Vec<f64>,
    pub active_valid: Vec<f64>,
    pub active_index: Vec<f64>,
    pub latest_update: TensorActuatorUpdateValues,
    pub last_change: TensorActuatorChangeValues,
    pub last_completion: TensorActuatorCompletionValues,
}
