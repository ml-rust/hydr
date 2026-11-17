//! Persistent original target literals in the shared actual gate order.

use crate::structures::tensor::actuator::TensorActuatorUpdate;
use crate::structures::{GateActuatorBatch, OperatingControlBatch, StructureError};
use numr::dtype::DType;
use numr::ops::ConditionalOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Debug)]
pub(crate) struct TensorControlledGateContext<R: Runtime<DType = DType>> {
    low_target: Tensor<R>,
    high_target: Tensor<R>,
}

impl<R: Runtime<DType = DType>> TensorControlledGateContext<R> {
    pub(crate) fn new(
        controls: &OperatingControlBatch<R>,
        actuators: &GateActuatorBatch<R>,
    ) -> Result<Self, StructureError> {
        let binding = super::binding::check(controls, actuators)?;
        Ok(Self {
            low_target: actuators.context.tensor(&binding.low_targets)?,
            high_target: actuators.context.tensor(&binding.high_targets)?,
        })
    }

    pub(crate) fn update(
        &self,
        controls: &OperatingControlBatch<R>,
        actuators: &GateActuatorBatch<R>,
        end_s: f64,
    ) -> Result<TensorActuatorUpdate<R>, StructureError>
    where
        R::Client: numr::ops::ConditionalOps<R>,
    {
        let context = &actuators.context;
        let client = &context.structures.client;
        let target =
            client.where_cond(&controls.state.high, &self.high_target, &self.low_target)?;
        let issued = context.constant(end_s)?;
        Ok(TensorActuatorUpdate {
            valid: context.one.clone(),
            target,
            available: issued.clone(),
            issued,
        })
    }
}
