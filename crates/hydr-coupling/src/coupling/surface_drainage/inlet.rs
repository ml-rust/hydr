//! Inlet/manhole exchange definitions, configuration, and bidirectional discharge closure.

use super::authority::DrainageControlClaim;
use super::error::SurfaceDrainageError;

/// Physical manhole or catch-basin inlet linking one 2D surface cell to one SWMM drainage node.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceDrainageInlet {
    /// Unique inlet identifier.
    pub id: String,
    /// Active row-major 2D surface grid cell index.
    pub surface_cell: usize,
    /// Target SWMM drainage node identifier from the `.inp` file.
    pub drainage_node_id: String,
    /// Inlet rim / street crest elevation in meters above the shared vertical datum.
    pub rim_elevation_m: f64,
    /// Effective weir crest perimeter in meters (`> 0`).
    pub weir_crest_length_m: f64,
    /// Weir discharge coefficient (`> 0`, SI `m^(1/2)/s`, e.g. `1.70`).
    pub weir_cd: f64,
    /// Clear orifice opening area in square meters (`> 0`).
    pub orifice_area_m2: f64,
    /// Orifice discharge coefficient (`> 0`, dimensionless, e.g. `0.62`).
    pub orifice_cd: f64,
    /// Whether pressurized sewer head above rim and surface stage can surcharge onto the surface.
    pub allow_surcharge_to_surface: bool,
}

impl SurfaceDrainageInlet {
    /// Checks geometric and coefficient positivity and finiteness.
    pub(crate) fn validate(&self) -> Result<(), SurfaceDrainageError> {
        if self.id.trim().is_empty() {
            return Err(SurfaceDrainageError::input("inlet id must not be empty"));
        }
        if self.drainage_node_id.trim().is_empty() {
            return Err(SurfaceDrainageError::input(format!(
                "inlet `{}` drainage_node_id must not be empty",
                self.id
            )));
        }
        if !self.rim_elevation_m.is_finite() {
            return Err(SurfaceDrainageError::input(format!(
                "inlet `{}` rim_elevation_m must be finite",
                self.id
            )));
        }
        for (name, value) in [
            ("weir_crest_length_m", self.weir_crest_length_m),
            ("weir_cd", self.weir_cd),
            ("orifice_area_m2", self.orifice_area_m2),
            ("orifice_cd", self.orifice_cd),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(SurfaceDrainageError::input(format!(
                    "inlet `{}` {name} must be finite and positive, got {value}",
                    self.id
                )));
            }
        }
        Ok(())
    }
}

/// Numerical coupling controls for [`super::SurfaceDrainageBatch`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceDrainageConfig {
    /// Maximum common step duration in seconds.
    pub max_step_s: f64,
    /// Minimum retried step duration in seconds.
    pub min_step_s: f64,
    /// Maximum step-halving retries per `trial_to` call.
    pub max_retries: usize,
    /// Maximum midpoint-head Picard iterations per attempted step.
    pub max_iterations: usize,
    /// Head convergence tolerance in meters.
    pub head_tolerance_m: f64,
    /// Exchange volume convergence tolerance in cubic meters.
    pub volume_tolerance_m3: f64,
    /// Gravitational acceleration in `m/s²` (default `9.81`).
    pub gravity: f64,
}

impl Default for SurfaceDrainageConfig {
    fn default() -> Self {
        Self {
            max_step_s: 30.0,
            min_step_s: 0.25,
            max_retries: 8,
            max_iterations: 16,
            head_tolerance_m: 1e-4,
            volume_tolerance_m3: 1e-5,
            gravity: 9.81,
        }
    }
}

impl SurfaceDrainageConfig {
    pub(crate) fn validate(&self) -> Result<(), SurfaceDrainageError> {
        if !self.max_step_s.is_finite() || self.max_step_s <= 0.0 {
            return Err(SurfaceDrainageError::input(format!(
                "max_step_s must be finite and positive, got {}",
                self.max_step_s
            )));
        }
        if !self.min_step_s.is_finite()
            || self.min_step_s <= 0.0
            || self.min_step_s > self.max_step_s
        {
            return Err(SurfaceDrainageError::input(format!(
                "min_step_s ({}) must be in (0, max_step_s ({})]",
                self.min_step_s, self.max_step_s
            )));
        }
        if self.max_iterations == 0 {
            return Err(SurfaceDrainageError::input(
                "max_iterations must be at least 1",
            ));
        }
        if !self.head_tolerance_m.is_finite() || self.head_tolerance_m <= 0.0 {
            return Err(SurfaceDrainageError::input(format!(
                "head_tolerance_m must be finite and positive, got {}",
                self.head_tolerance_m
            )));
        }
        if !self.volume_tolerance_m3.is_finite() || self.volume_tolerance_m3 <= 0.0 {
            return Err(SurfaceDrainageError::input(format!(
                "volume_tolerance_m3 must be finite and positive, got {}",
                self.volume_tolerance_m3
            )));
        }
        if !self.gravity.is_finite() || self.gravity <= 0.0 {
            return Err(SurfaceDrainageError::input(format!(
                "gravity must be finite and positive, got {}",
                self.gravity
            )));
        }
        Ok(())
    }
}

/// Complete static setup for a coupled 2D surface / SWMM drainage coordinator.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceDrainageSetup {
    /// Declared vertical datum shared by the 2D surface DEM and the SWMM drainage network.
    pub vertical_datum: String,
    /// Numerical coupling tolerances and retry limits.
    pub config: SurfaceDrainageConfig,
    /// Physical inlets connecting surface cells to SWMM drainage nodes.
    pub inlets: Vec<SurfaceDrainageInlet>,
    /// Explicit control-authority claims checked against native SWMM `[CONTROLS]`.
    pub control_claims: Vec<DrainageControlClaim>,
}

/// Evaluates the signed inlet exchange discharge in `m³/s`.
///
/// Positive discharge flows from the 2D surface into the SWMM drainage node (weir/orifice
/// capture). Negative discharge flows from a surcharged drainage node onto the 2D surface
/// when `inlet.allow_surcharge_to_surface` is `true`.
pub fn inlet_discharge_m3_per_s(
    inlet: &SurfaceDrainageInlet,
    surface_stage_m: f64,
    drainage_head_m: f64,
    gravity: f64,
) -> f64 {
    let rim = inlet.rim_elevation_m;
    if surface_stage_m > rim && surface_stage_m > drainage_head_m {
        let h_s = surface_stage_m - rim;
        let dh = surface_stage_m - drainage_head_m.max(rim);
        let q_weir = inlet.weir_cd * inlet.weir_crest_length_m * h_s * dh.sqrt();
        let q_orifice = inlet.orifice_cd * inlet.orifice_area_m2 * (2.0 * gravity * dh).sqrt();
        return q_weir.min(q_orifice);
    }
    if inlet.allow_surcharge_to_surface
        && drainage_head_m > rim
        && drainage_head_m > surface_stage_m
    {
        let dh = drainage_head_m - surface_stage_m.max(rim);
        let q_surcharge = inlet.orifice_cd * inlet.orifice_area_m2 * (2.0 * gravity * dh).sqrt();
        return -q_surcharge;
    }
    0.0
}
