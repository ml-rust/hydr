//! Explicit actual pump operation under the separately declared ideal response model.

use super::PumpResponseModel;
use crate::structures::{ControlLatch, StructureOperatingPoint};

/// Projected actual operation without an inferred hydraulic or accepted-transfer result.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlledPumpOutput {
    /// Index in the complete original structure model.
    pub element: usize,
    /// Original sampled controller identity.
    pub controller_id: String,
    /// Original actual pump identity.
    pub structure_id: String,
    /// Original accepted control time, preserving its F64 bits.
    pub time_s: f64,
    /// Original accepted interval count from the complete control continuation.
    pub accepted_steps: usize,
    /// Explicit physical response approximation.
    pub model: PumpResponseModel,
    /// Accepted sampled latch selecting the original operation literal.
    pub latch: ControlLatch,
    /// Actual stopped isolation or running speed. Stopped output contains no fabricated speed.
    pub operating_point: StructureOperatingPoint,
}
