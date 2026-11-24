//! Configuration, conservation ledger, and portable checkpoint types for `SurfaceSolverF32`.

use super::muscl::SurfaceF32SpatialOrder;
use crate::contract::EngineError;

/// Stable snapshot schema identifier for `SurfaceSolverF32`.
pub const SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA: &str = "hydr-f32-checkpoint-1";

/// Stable numerical contract identifier for `SurfaceSolverF32`.
pub const SURFACE_SOLVER_F32_CONTRACT_VERSION: &str = "hydr-swe-f32-numr-1";

/// Numerical controls for the opt-in `F32` 2D shallow-water solver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSolverF32Config {
    /// Gravitational acceleration in `m/s²` (default `9.81`).
    pub gravity: f32,
    /// Courant-Friedrichs-Lewy safety factor in `(0, 0.5]` (default `0.45`).
    pub cfl: f32,
    /// Wet/dry velocity cutoff depth in meters (default `1e-4`).
    pub dry_depth_m: f32,
    /// Maximum time step in seconds (default `5.0`).
    pub max_step_s: f64,
    /// Minimum retried time step in seconds (default `1e-5`).
    pub min_step_s: f64,
    /// Maximum step-halving retries per step (default `16`).
    pub max_retries: usize,
    /// Spatial reconstruction order (default `FirstOrder`; opt into `SecondOrderMuscl` for sharper wave fronts).
    pub spatial_order: SurfaceF32SpatialOrder,
}

impl Default for SurfaceSolverF32Config {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            cfl: 0.45,
            dry_depth_m: 1e-4,
            max_step_s: 5.0,
            min_step_s: 1e-5,
            max_retries: 16,
            spatial_order: SurfaceF32SpatialOrder::FirstOrder,
        }
    }
}

impl SurfaceSolverF32Config {
    pub(crate) fn validate(&self) -> Result<(), EngineError> {
        let reject = |msg: String| EngineError::rejected("surface solver f32 config", msg);
        if !self.gravity.is_finite() || self.gravity <= 0.0 {
            return Err(reject(format!(
                "gravity must be finite and positive, got {}",
                self.gravity
            )));
        }
        if !self.cfl.is_finite() || self.cfl <= 0.0 || self.cfl > 0.5 {
            return Err(reject(format!("cfl must be in (0, 0.5], got {}", self.cfl)));
        }
        if !self.dry_depth_m.is_finite() || self.dry_depth_m <= 0.0 {
            return Err(reject(format!(
                "dry_depth_m must be finite and positive, got {}",
                self.dry_depth_m
            )));
        }
        if !self.max_step_s.is_finite() || self.max_step_s <= 0.0 {
            return Err(reject(format!(
                "max_step_s must be finite and positive, got {}",
                self.max_step_s
            )));
        }
        if !self.min_step_s.is_finite()
            || self.min_step_s <= 0.0
            || self.min_step_s > self.max_step_s
        {
            return Err(reject(format!(
                "min_step_s ({}) must be in (0, max_step_s ({})]",
                self.min_step_s, self.max_step_s
            )));
        }
        Ok(())
    }
}

/// Cumulative SI water-balance ledger tracked in host `f64` for `SurfaceSolverF32`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSolverF32Ledger {
    /// Initial stored water volume (`m³`).
    pub initial_storage_m3: f64,
    /// Current stored water volume (`m³`).
    pub final_storage_m3: f64,
    /// Cumulative rainfall volume added (`m³`).
    pub rain_m3: f64,
    /// Cumulative evaporation volume removed (`m³`).
    pub evaporation_m3: f64,
    /// Cumulative prescribed withdrawal volume removed (`m³`).
    pub withdrawal_m3: f64,
    /// Cumulative Modified Green-Ampt soil infiltration removed (`m³`).
    pub infiltration_m3: f64,
    /// Cumulative open-boundary inflow added (`m³`).
    pub boundary_inflow_m3: f64,
    /// Cumulative open-boundary outflow removed (`m³`).
    pub boundary_outflow_m3: f64,
    /// Conservation residual `(final - initial) - (rain - evap - withdrawal - infiltration + boundary_in - boundary_out)` (`m³`).
    pub residual_m3: f64,
}

impl SurfaceSolverF32Ledger {
    pub(super) fn zero(initial_storage_m3: f64) -> Self {
        Self {
            initial_storage_m3,
            final_storage_m3: initial_storage_m3,
            rain_m3: 0.0,
            evaporation_m3: 0.0,
            withdrawal_m3: 0.0,
            infiltration_m3: 0.0,
            boundary_inflow_m3: 0.0,
            boundary_outflow_m3: 0.0,
            residual_m3: 0.0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn record_step(
        &mut self,
        rain_m3: f64,
        evap_m3: f64,
        withdrawal_m3: f64,
        infil_m3: f64,
        boundary_in_m3: f64,
        boundary_out_m3: f64,
        final_storage_m3: f64,
    ) {
        self.rain_m3 += rain_m3;
        self.evaporation_m3 += evap_m3;
        self.withdrawal_m3 += withdrawal_m3;
        self.infiltration_m3 += infil_m3;
        self.boundary_inflow_m3 += boundary_in_m3;
        self.boundary_outflow_m3 += boundary_out_m3;
        self.final_storage_m3 = final_storage_m3;
        let net = self.rain_m3 - self.evaporation_m3 - self.withdrawal_m3 - self.infiltration_m3
            + self.boundary_inflow_m3
            - self.boundary_outflow_m3;
        self.residual_m3 = (final_storage_m3 - self.initial_storage_m3) - net;
    }
}

/// Portable complete checkpoint for `SurfaceSolverF32`.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceSolverF32SnapshotData {
    /// Snapshot schema (`SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA`).
    pub schema: String,
    /// Contract version (`SURFACE_SOLVER_F32_CONTRACT_VERSION`).
    pub contract: String,
    /// Accepted simulation time (`s`).
    pub time_s: f64,
    /// Number of accepted steps.
    pub accepted_steps: u64,
    /// Row-major cell depths (`m`, `f32`).
    pub h_m: Vec<f32>,
    /// Row-major cell x-discharges per width (`m²/s`, `f32`).
    pub qx_m2_per_s: Vec<f32>,
    /// Row-major cell y-discharges per width (`m²/s`, `f32`).
    pub qy_m2_per_s: Vec<f32>,
    /// Cumulative Modified Green-Ampt infiltration per cell (`m`, `f32`), when soil is attached.
    pub cumulative_infiltration_m: Option<Vec<f32>>,
    /// Cumulative water-balance ledger.
    pub ledger: SurfaceSolverF32Ledger,
}

impl SurfaceSolverF32SnapshotData {
    pub(super) fn validate(&self, expected_count: usize) -> Result<(), EngineError> {
        let reject = |msg: String| EngineError::rejected("surface solver f32 snapshot", msg);
        if self.schema != SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA {
            return Err(reject(format!(
                "schema `{}` does not match `{SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA}`",
                self.schema
            )));
        }
        if self.contract != SURFACE_SOLVER_F32_CONTRACT_VERSION {
            return Err(reject(format!(
                "contract `{}` does not match `{SURFACE_SOLVER_F32_CONTRACT_VERSION}`",
                self.contract
            )));
        }
        if self.h_m.len() != expected_count
            || self.qx_m2_per_s.len() != expected_count
            || self.qy_m2_per_s.len() != expected_count
        {
            return Err(reject(format!(
                "snapshot cell length mismatch (expected {expected_count})"
            )));
        }
        if self.h_m.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self.qx_m2_per_s.iter().any(|v| !v.is_finite())
            || self.qy_m2_per_s.iter().any(|v| !v.is_finite())
        {
            return Err(reject(
                "snapshot contains non-finite or negative depth values".into(),
            ));
        }
        Ok(())
    }
}
