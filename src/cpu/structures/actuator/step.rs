//! One bounded accepted actuator interval without prior-step ownership.

use super::{CpuActuatorForward, CpuActuatorPosition, CpuActuatorState, CpuActuatorUpdate};
use crate::cpu::CpuTensor;
use crate::structures::GateActuatorEvent;

#[derive(Clone, Debug)]
pub(crate) struct CpuActuatorStep {
    pub start: f64,
    pub end: f64,
    pub event: Option<GateActuatorEvent>,
    pub before: CpuActuatorState,
    pub after: CpuActuatorState,
    pub left: CpuActuatorPosition,
    pub forward: CpuActuatorForward,
    pub update: CpuActuatorUpdate,
    pub completed: CpuTensor,
    pub changed: CpuTensor,
}
