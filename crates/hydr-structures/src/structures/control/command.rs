//! Explicit commands without hydraulic consumption or actuator response.

/// Current side of one sampled hysteresis relay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlLatch {
    /// Lower relay state.
    Low,
    /// Upper relay state.
    High,
}
/// Requested device operation. Gate targets never assert actual opening.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ControlCommand {
    /// Explicit stopped pump isolation command.
    PumpStoppedIsolated,
    /// Explicit running speed command without inferred response dynamics.
    PumpRunning {
        /// Positive supported speed ratio.
        speed_ratio: f64,
    },
    /// Desired relative gate position, distinct from actual aperture opening.
    GateTarget {
        /// Requested opening in metres.
        opening_m: f64,
    },
}
/// Explicit command projection for one controlled structure.
#[derive(Clone, Debug, PartialEq)]
pub struct ControlCommandOutput {
    /// Complete structure-batch index.
    pub element: usize,
    /// Original controller identity.
    pub controller_id: String,
    /// Original structure identity.
    pub structure_id: String,
    /// Supported command time in seconds.
    pub time_s: f64,
    /// Current relay latch.
    pub latch: ControlLatch,
    /// Selected original command.
    pub command: ControlCommand,
}
