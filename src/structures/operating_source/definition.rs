//! Explicit source ownership and borrowed fresh response frames.

use crate::structures::{ControlledPumpBatch, GateActuatorBatch, GateTrajectoryBatch};

/// One explicit writer for a nonempty ordered subset of the complete actual structure batch.
#[derive(Clone, Debug)]
pub enum OperatingSourceBinding<'a> {
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
        trajectory: &'a GateTrajectoryBatch,
    },
    /// Actual gate positions supplied through a fresh accepted frame at each query.
    Actuated {
        /// Unique nonempty source identity.
        source_id: String,
        /// Only this batch's immutable context is captured during registration.
        actuator: &'a GateActuatorBatch,
    },
    /// Explicit ideal instantaneous pump response supplied through a fresh accepted frame.
    ControlledPump {
        /// Unique nonempty source identity.
        source_id: String,
        /// Only immutable control and response definitions are captured during registration.
        pump: &'a ControlledPumpBatch,
    },
}
/// Fresh accepted response borrowed for one exact numerical query time.
#[derive(Clone, Copy, Debug)]
pub enum OperatingSourceFrame<'a> {
    /// Accepted actual gate response without implicit advancement or interpolation.
    ActuatedGate {
        /// Registered source identity in dynamic-group registration order.
        source_id: &'a str,
        /// Accepted ideal finite-rate response.
        actuator: &'a GateActuatorBatch,
    },
    /// Accepted sampled pump response under its explicit instantaneous assumption.
    ControlledPump {
        /// Registered source identity in dynamic-group registration order.
        source_id: &'a str,
        /// Accepted sampled commands and immutable response assumptions.
        pump: &'a ControlledPumpBatch,
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
