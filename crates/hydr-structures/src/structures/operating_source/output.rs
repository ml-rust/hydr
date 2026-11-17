//! Mandatory source provenance and gate-opening uncertainty paired with constitutive output.

use crate::structures::{
    ControlLatch, GateActuatorPhase, GateActuatorPosition, GateActuatorSupport, PumpResponseModel,
    StructureOperatingPoint, StructureOutput,
};

/// An instantaneous hydraulic result with its responsible operating source.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatingSourceOutput {
    /// Hydraulic law evaluated from the admitted actual operation and supplied heads.
    pub hydraulic: StructureOutput,
    /// Source ownership, point support, and source-specific operation or opening evidence.
    pub source: OperatingSourceReceipt,
}
/// Identity and original point time attached to every hydraulic output row.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatingSourceReceipt {
    /// Registered unique source identity.
    pub source_id: String,
    /// Original complete structure index.
    pub element: usize,
    /// Original complete structure identity.
    pub structure_id: String,
    /// Original query time, preserving its F64 bits.
    pub time_s: f64,
    /// Original source semantics, actual operation, and applicable opening evidence.
    pub detail: OperatingSourceDetail,
}
/// Source-specific evidence. Position bounds imply no ideal-flow or integrated-volume enclosure.
#[derive(Clone, Debug, PartialEq)]
pub enum OperatingSourceDetail {
    /// Original manual packet, without an inferred measurement-error model.
    Manual,
    /// Original-time prescribed gate support and interpolation error.
    PrescribedGate {
        /// Solvr absolute opening-error enclosure in metres.
        absolute_opening_error_m: f64,
        /// Original right-sided segment, except at the final knot.
        segment_index: usize,
        /// Exact original knot index when represented time equality holds.
        exact_knot_index: Option<usize>,
        /// Original zero knot or zero hold certifies closure at this point.
        closed_by_definition: bool,
    },
    /// Actual accepted ideal actuator response with complete positional uncertainty.
    ActuatedGate {
        /// Original actuator identity in the registered ordered subset.
        actuator_id: String,
        /// Accepted original interval count at this frame.
        accepted_steps: usize,
        /// Original frame accepted time, preserving its bits independently of the query time.
        accepted_time_s: f64,
        /// Admitted actual opening center, interval, and absolute error in metres.
        position: GateActuatorPosition,
        /// Admitted current motion/completion phase.
        phase: GateActuatorPhase,
        /// Canonical closed, certainly positive, or unresolved closure support.
        support: GateActuatorSupport,
    },
    /// Accepted sampled command under an explicit ideal instantaneous response model.
    ControlledPump {
        /// Original controller identity in the registered ordered pump subset.
        controller_id: String,
        /// Accepted original control interval count at this frame.
        accepted_steps: usize,
        /// Original frame accepted time, preserving its bits independently of the query time.
        accepted_time_s: f64,
        /// Accepted sampled relay latch selecting the original command.
        latch: ControlLatch,
        /// Explicit assumed response model, without inferred motor dynamics.
        response_model: PumpResponseModel,
        /// Actual stopped isolation or original running speed under the declared model.
        operating_point: StructureOperatingPoint,
    },
}
