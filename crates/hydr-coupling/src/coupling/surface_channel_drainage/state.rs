//! Setup, diagnostics, and portable 3-way checkpoint data for `SurfaceChannelDrainageBatch`.

use crate::coupling::{
    DrainageControlClaim, SurfaceChannelSnapshotData, SurfaceChannelStepDiagnostics,
    SurfaceDrainageConfig, SurfaceDrainageInlet, SurfaceDrainageInletTransfer,
};
use crate::drainage::SwmmDrainageSnapshotData;

/// Stable contract version of the 3-way surface-channel-drainage coordinator.
pub const SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION: &str = "hydr.surface-channel-drainage/1";

/// Stable snapshot schema of the 3-way surface-channel-drainage coordinator.
pub const SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA: &str =
    "hydr.surface-channel-drainage-snapshot/1";

/// Static setup for coupling a [`crate::coupling::SurfaceChannelBatch`] with a
/// [`crate::drainage::SwmmDrainage`] network.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelDrainageSetup {
    /// Shared vertical datum identifier; must match `SwmmDrainage::definition().vertical_datum()`.
    pub vertical_datum: String,
    /// Bidirectional surface↔drainage inlets. Must be non-empty with unique ids and `(cell, node)` pairs.
    pub inlets: Vec<SurfaceDrainageInlet>,
    /// Exclusive control-authority claims for SWMM links actuated by `hydr`.
    pub control_claims: Vec<DrainageControlClaim>,
    /// Iteration tolerances, gravity, and step limits for the outer drainage exchange loop.
    pub config: SurfaceDrainageConfig,
}

/// Complete step receipt for one accepted 3-way coupled interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelDrainageStepDiagnostics {
    /// Common interval start time in seconds.
    pub start_s: f64,
    /// Common interval end time in seconds.
    pub end_s: f64,
    /// Common step duration `end_s - start_s` in seconds.
    pub dt_s: f64,
    /// Number of outer step-shortening retries performed.
    pub retries: usize,
    /// Number of outer midpoint-head iterations performed on the accepted step.
    pub iterations: usize,
    /// Final outer head residual across surface inlets and SWMM nodes in metres.
    pub max_head_residual_m: f64,
    /// Final outer inlet transfer volume residual in cubic metres.
    pub max_volume_residual_m3: f64,
    /// Nested 2D surface / 1D channel / pool / structure step diagnostics on `[start_s, end_s]`.
    pub surface_channel: SurfaceChannelStepDiagnostics,
    /// Per-inlet signed transfer records (`+` surface→drainage, `-` drainage→surface surcharge).
    pub inlet_transfers: Vec<SurfaceDrainageInletTransfer>,
}

/// Portable complete 3-way checkpoint capturing surface, channels/pools/structures, and SWMM drainage.
#[derive(Clone, Debug)]
pub struct SurfaceChannelDrainageSnapshotData {
    /// Schema identifier (`SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA`).
    pub schema: String,
    /// Contract identifier (`SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION`).
    pub contract: String,
    /// Vertical datum identifier.
    pub vertical_datum: String,
    /// Common accepted time in seconds.
    pub time_s: f64,
    /// Number of accepted 3-way coupled steps.
    pub accepted_steps: u64,
    /// Cumulative signed transfer volume (`m³`) per inlet in setup order.
    pub cumulative_inlet_transfers_m3: Vec<f64>,
    /// Nested 2D surface + 1D channel/pool/structure checkpoint.
    pub surface_channel: SurfaceChannelSnapshotData,
    /// Nested pinned EPA SWMM v5.2.4 piped drainage checkpoint.
    pub drainage: SwmmDrainageSnapshotData,
}
