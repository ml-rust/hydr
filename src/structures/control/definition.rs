//! Immutable original sampled-relay definitions.

use super::{ControlCommand, ControlStageSignal};

/// Explicit two-threshold sampled qualification model for one actual device.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatingControlDefinition {
    /// Unique controller identity.
    pub controller_id: String,
    /// Unique actual controlled structure identity.
    pub structure_id: String,
    /// Explicit external observation support.
    pub signal: ControlStageSignal,
    /// Inclusive stage threshold for leaving the high latch, in metres.
    pub lower_stage_m: f64,
    /// Inclusive stage threshold for leaving the low latch, in metres.
    pub upper_stage_m: f64,
    /// Required sampled high qualification duration in seconds.
    pub high_qualification_delay_s: f64,
    /// Required sampled low qualification duration in seconds.
    pub low_qualification_delay_s: f64,
    /// Original low-latch command.
    pub low_command: ControlCommand,
    /// Original high-latch command.
    pub high_command: ControlCommand,
    /// Source and physical operating-envelope evidence.
    pub provenance: String,
}
