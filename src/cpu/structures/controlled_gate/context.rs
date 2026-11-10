//! Persistent original target literals in the shared actual gate order.

use crate::cpu::CpuTensor;
use crate::cpu::structures::actuator::CpuActuatorUpdate;
use crate::structures::{GateActuatorBatch, OperatingControlBatch, StructureError};
use numr::ops::ConditionalOps;

#[derive(Debug)]
pub(crate) struct CpuControlledGateContext {
    low_target: CpuTensor,
    high_target: CpuTensor,
}

impl CpuControlledGateContext {
    pub(crate) fn new(
        controls: &OperatingControlBatch,
        actuators: &GateActuatorBatch,
    ) -> Result<Self, StructureError> {
        let binding = super::binding::check(controls, actuators)?;
        Ok(Self {
            low_target: actuators.context.tensor(&binding.low_targets)?,
            high_target: actuators.context.tensor(&binding.high_targets)?,
        })
    }

    pub(crate) fn update(
        &self,
        controls: &OperatingControlBatch,
        actuators: &GateActuatorBatch,
        end_s: f64,
    ) -> Result<CpuActuatorUpdate, StructureError> {
        let context = &actuators.context;
        let client = &context.structures.client;
        let target =
            client.where_cond(&controls.state.high, &self.high_target, &self.low_target)?;
        let issued = context.constant(end_s)?;
        Ok(CpuActuatorUpdate {
            valid: context.one.clone(),
            target,
            available: issued.clone(),
            issued,
        })
    }
}
