//! Transfer records, step diagnostics, and portable checkpoints for `SurfaceDrainageBatch`.

use crate::drainage::SwmmDrainageSnapshotData;
use crate::solver::SnapshotData as SurfaceSnapshotData;

/// Stable snapshot schema identifier for `SurfaceDrainageBatch`.
pub const SURFACE_DRAINAGE_SNAPSHOT_SCHEMA: &str = "hydr-surface-drainage-checkpoint-1";

/// Stable contract version for `SurfaceDrainageBatch`.
pub const SURFACE_DRAINAGE_CONTRACT_VERSION: &str = "hydr-surface-drainage-numr-cpu-1";

/// Accepted signed water transfer across one inlet during one coupled step.
///
/// Positive volume leaves the 2D surface cell and enters the SWMM drainage node.
/// Negative volume leaves a surcharged SWMM drainage node and enters the 2D surface cell.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceDrainageInletTransfer {
    /// Inlet identifier.
    pub inlet_id: String,
    /// Linked 2D surface cell index.
    pub surface_cell: usize,
    /// Linked SWMM drainage node index.
    pub drainage_node_index: usize,
    /// Linked SWMM drainage node identifier.
    pub drainage_node_id: String,
    /// Exact signed transfer volume accepted by both components during the step (`m³`).
    pub transfer_volume_m3: f64,
    /// Average signed exchange rate during the step (`transfer_volume_m3 / dt_s`, `m³/s`).
    pub average_rate_m3_per_s: f64,
    /// Surface stage at the end of the step (`m`).
    pub end_surface_stage_m: f64,
    /// Drainage hydraulic head at the end of the step (`m`).
    pub end_drainage_head_m: f64,
}

/// Convergence and accounting summary of one coupled surface-drainage step.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceDrainageStepDiagnostics {
    /// Accepted step start time (`s`).
    pub start_s: f64,
    /// Accepted step end time (`s`).
    pub end_s: f64,
    /// Accepted step duration (`s`).
    pub dt_s: f64,
    /// Step-halving retries needed before convergence.
    pub retries: usize,
    /// Midpoint-head fixed-point iterations used on the accepted attempt.
    pub iterations: usize,
    /// Final head residual across all inlets (`m`).
    pub max_head_residual_m: f64,
    /// Final volume residual across all inlets (`m³`).
    pub max_volume_residual_m3: f64,
    /// Per-inlet signed transfer records.
    pub transfers: Vec<SurfaceDrainageInletTransfer>,
}

/// Portable complete checkpoint for a coupled 2D surface / SWMM drainage run.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceDrainageSnapshotData {
    /// Snapshot schema identifier (`SURFACE_DRAINAGE_SNAPSHOT_SCHEMA`).
    pub schema: String,
    /// Contract version (`SURFACE_DRAINAGE_CONTRACT_VERSION`).
    pub contract: String,
    /// Declared shared vertical datum.
    pub vertical_datum: String,
    /// Accepted common simulation time (`s`).
    pub time_s: f64,
    /// Number of accepted coupled steps.
    pub accepted_steps: u64,
    /// Cumulative signed transfer volume (`m³`) per inlet in setup order.
    pub cumulative_inlet_transfers_m3: Vec<f64>,
    /// Complete 2D surface solver checkpoint.
    pub surface: SurfaceSnapshotData,
    /// Complete SWMM v5.2.4 drainage worker checkpoint.
    pub drainage: SwmmDrainageSnapshotData,
}
