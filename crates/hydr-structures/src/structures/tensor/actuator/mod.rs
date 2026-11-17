//! Persistent batched actual gate actuator state and flat bounded evidence.

mod context;
mod forward;
mod position;
pub(crate) mod record;
mod state;
mod step;
pub(crate) use context::TensorActuatorContext;
pub(crate) use forward::{
    TensorActuatorEndpoint, TensorActuatorEndpointValues, TensorActuatorFma,
    TensorActuatorFmaValues, TensorActuatorForward, TensorActuatorForwardValues,
    TensorActuatorMotion, TensorActuatorMotionValues,
};
pub(crate) use position::{
    TensorActuatorAnchor, TensorActuatorAnchorValues, TensorActuatorArrival,
    TensorActuatorArrivalValues, TensorActuatorPosition, TensorActuatorPositionValues,
};
pub(crate) use state::{
    TensorActuatorChange, TensorActuatorChangeValues, TensorActuatorCompletion,
    TensorActuatorCompletionValues, TensorActuatorState, TensorActuatorStateValues,
    TensorActuatorUpdate, TensorActuatorUpdateValues,
};
pub(crate) use step::TensorActuatorStep;

pub(crate) mod import_values;
