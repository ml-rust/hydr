//! Opt-in coastal & meteorological momentum forcing for certified `F64` `Solver<R>`.

use crate::contract::EngineError;

/// Opt-in coastal & meteorological momentum forcing applied on wet cells of [`super::Solver`].
///
/// Default has all forcing terms zero (zero overhead, bit-identical to unforced `Solver`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CoastalForcing {
    /// Uniform 10 m wind x-component (`m/s`).
    pub wind_x_m_per_s: f64,
    /// Uniform 10 m wind y-component (`m/s`).
    pub wind_y_m_per_s: f64,
    /// Dimensionless wind drag coefficient `C_d` (default `1.5e-3`).
    pub wind_drag_coeff: f64,
    /// Coriolis frequency `f = 2 * Omega * sin(lat)` (`1/s`, default `0.0`).
    pub coriolis_f_per_s: f64,
    /// Atmospheric pressure gradient `dP/dx` (`Pa/m`, default `0.0`).
    pub dp_dx_pa_per_m: f64,
    /// Atmospheric pressure gradient `dP/dy` (`Pa/m`, default `0.0`).
    pub dp_dy_pa_per_m: f64,
    /// Water density (`kg/m³`, default `1000.0`).
    pub water_density_kg_per_m3: f64,
    /// Air density (`kg/m³`, default `1.225`).
    pub air_density_kg_per_m3: f64,
}

impl Default for CoastalForcing {
    fn default() -> Self {
        Self {
            wind_x_m_per_s: 0.0,
            wind_y_m_per_s: 0.0,
            wind_drag_coeff: 1.5e-3,
            coriolis_f_per_s: 0.0,
            dp_dx_pa_per_m: 0.0,
            dp_dy_pa_per_m: 0.0,
            water_density_kg_per_m3: 1000.0,
            air_density_kg_per_m3: 1.225,
        }
    }
}

impl CoastalForcing {
    /// Returns true if any coastal forcing term is nonzero.
    pub fn is_active(&self) -> bool {
        self.wind_x_m_per_s != 0.0
            || self.wind_y_m_per_s != 0.0
            || self.coriolis_f_per_s != 0.0
            || self.dp_dx_pa_per_m != 0.0
            || self.dp_dy_pa_per_m != 0.0
    }

    /// Validates all forcing parameters.
    pub fn validate(&self) -> Result<(), EngineError> {
        let reject = |msg: String| EngineError::rejected("coastal forcing", msg);
        for (name, v) in [
            ("wind_x_m_per_s", self.wind_x_m_per_s),
            ("wind_y_m_per_s", self.wind_y_m_per_s),
            ("coriolis_f_per_s", self.coriolis_f_per_s),
            ("dp_dx_pa_per_m", self.dp_dx_pa_per_m),
            ("dp_dy_pa_per_m", self.dp_dy_pa_per_m),
        ] {
            if !v.is_finite() {
                return Err(reject(format!("{name} must be finite, got {v}")));
            }
        }
        if !self.wind_drag_coeff.is_finite() || self.wind_drag_coeff < 0.0 {
            return Err(reject(format!(
                "wind_drag_coeff must be finite and >= 0, got {}",
                self.wind_drag_coeff
            )));
        }
        if !self.water_density_kg_per_m3.is_finite() || self.water_density_kg_per_m3 <= 0.0 {
            return Err(reject(format!(
                "water_density_kg_per_m3 must be positive, got {}",
                self.water_density_kg_per_m3
            )));
        }
        if !self.air_density_kg_per_m3.is_finite() || self.air_density_kg_per_m3 < 0.0 {
            return Err(reject(format!(
                "air_density_kg_per_m3 must be >= 0, got {}",
                self.air_density_kg_per_m3
            )));
        }
        Ok(())
    }
}
