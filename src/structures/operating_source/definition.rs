//! Explicit source ownership and borrowed fresh response frames.

use crate::structures::{ControlledPumpBatch, GateActuatorBatch, GateTrajectoryBatch};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// One explicit writer for a nonempty ordered subset of the complete actual structure batch.
#[derive(Clone, Debug)]
pub enum OperatingSourceBinding<'a, R: Runtime<DType = DType>> {
    /// Original manual operation packets supplied afresh at every query.
    Manual {
        /// Unique nonempty source identity, retained in registration order.
        source_id: String,
        /// Nonempty unique original element identities in complete structure order.
        structure_ids: Vec<String>,
    },
    /// Immutable prescribed gate trajectories, restricted to their actual assigned subset.
    Prescribed {
        /// Unique nonempty source identity.
        source_id: String,
        /// Complete immutable trajectory binding cloned by the registry.
        trajectory: &'a GateTrajectoryBatch<R>,
    },
    /// Actual gate positions supplied through a fresh accepted frame at each query.
    Actuated {
        /// Unique nonempty source identity.
        source_id: String,
        /// Only this batch's immutable context is captured during registration.
        actuator: &'a GateActuatorBatch<R>,
    },
    /// Explicit ideal instantaneous pump response supplied through a fresh accepted frame.
    ControlledPump {
        /// Unique nonempty source identity.
        source_id: String,
        /// Only immutable control and response definitions are captured during registration.
        pump: &'a ControlledPumpBatch<R>,
    },
}
/// Fresh accepted response borrowed for one exact numerical query time.
#[derive(Debug)]
pub enum OperatingSourceFrame<'a, R: Runtime<DType = DType>> {
    /// Accepted actual gate response without implicit advancement or interpolation.
    ActuatedGate {
        /// Registered source identity in dynamic-group registration order.
        source_id: &'a str,
        /// Accepted ideal finite-rate response.
        actuator: &'a GateActuatorBatch<R>,
    },
    /// Accepted sampled pump response under its explicit instantaneous assumption.
    ControlledPump {
        /// Registered source identity in dynamic-group registration order.
        source_id: &'a str,
        /// Accepted sampled commands and immutable response assumptions.
        pump: &'a ControlledPumpBatch<R>,
    },
}
/// Closed common support of every bounded source. A singleton remains a valid point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OperatingSourceSupport {
    /// First common supported time in seconds.
    pub start_s: f64,
    /// Last common supported time in seconds.
    pub end_s: f64,
}

impl<R: Runtime<DType = DType>> Copy for OperatingSourceFrame<'_, R> {}
impl<R: Runtime<DType = DType>> Clone for OperatingSourceFrame<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}
