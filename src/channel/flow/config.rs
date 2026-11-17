use super::ChannelError;
use crate::contract::SolverConfig;

/// Spatial reconstruction for triangular channel cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelTriangularReconstruction {
    /// Piecewise constant reference method.
    FirstOrderReference,
    /// Slope-limited piecewise linear production method.
    LimitedLinear,
}

/// Numerical controls for a channel reach.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelConfig {
    /// Scalar endpoint compatibility controls.
    pub boundary: super::ChannelBoundaryConfig,
    /// Triangular spatial reconstruction. Rectangular reaches retain their existing method.
    pub triangular_reconstruction: ChannelTriangularReconstruction,
    /// Gravity in m/s².
    pub gravity: f64,
    /// Courant ceiling, greater than zero and at most 0.45.
    pub cfl: f64,
    /// Maximum accepted interval in seconds.
    pub max_step: f64,
    /// Minimum retry interval in seconds, excluding an exact shorter endpoint.
    pub min_step: f64,
    /// Maximum recalculations after an inadmissible trial.
    pub max_retries: usize,
    /// Absolute Riemann root-bracket width in metres.
    pub root_absolute_depth: f64,
    /// Relative Riemann root-bracket width.
    pub root_relative_depth: f64,
    /// Maximum bracket expansions and bisections per Riemann calculation.
    pub root_iterations: usize,
    /// Absolute water-flux bracket width tolerance in m³/s, separate from conservation tolerances.
    pub riemann_water_flux_absolute_m3_per_s: f64,
    /// Absolute momentum-flux bracket width tolerance in m⁴/s², separate from conservation tolerances.
    pub riemann_momentum_flux_absolute_m4_per_s2: f64,
    /// Relative flux bracket width tolerance for both conserved quantities.
    pub riemann_flux_relative: f64,
    /// Absolute water residual tolerance in m³.
    pub volume_absolute: f64,
    /// Relative water residual tolerance.
    pub volume_relative: f64,
    /// Absolute cell momentum residual tolerance in m⁴/s.
    pub momentum_absolute: f64,
    /// Relative cell momentum residual tolerance.
    pub momentum_relative: f64,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        let config = SolverConfig::default();
        Self {
            boundary: super::ChannelBoundaryConfig::default(),
            triangular_reconstruction: ChannelTriangularReconstruction::LimitedLinear,
            gravity: config.gravity,
            cfl: config.cfl,
            max_step: config.max_step,
            min_step: config.min_step,
            max_retries: config.max_retries,
            root_absolute_depth: config.root_absolute_depth,
            root_relative_depth: config.root_relative_depth,
            root_iterations: config.root_iterations,
            riemann_water_flux_absolute_m3_per_s: 1e-12,
            riemann_momentum_flux_absolute_m4_per_s2: 1e-12,
            riemann_flux_relative: 1e-12,
            volume_absolute: config.volume_absolute,
            volume_relative: config.volume_relative,
            momentum_absolute: 1e-9,
            momentum_relative: 1e-12,
        }
    }
}

impl ChannelConfig {
    pub(super) fn check(&self) -> Result<(), ChannelError> {
        self.boundary.check()?;
        for (name, value) in [
            ("gravity", self.gravity),
            ("cfl", self.cfl),
            ("max_step", self.max_step),
            ("min_step", self.min_step),
            ("root_absolute_depth", self.root_absolute_depth),
            ("volume_absolute", self.volume_absolute),
            ("momentum_absolute", self.momentum_absolute),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ChannelError::input(format!(
                    "{name}={value}: supply a finite positive value"
                )));
            }
        }
        for (name, value) in [
            ("root_relative_depth", self.root_relative_depth),
            (
                "riemann_water_flux_absolute_m3_per_s",
                self.riemann_water_flux_absolute_m3_per_s,
            ),
            (
                "riemann_momentum_flux_absolute_m4_per_s2",
                self.riemann_momentum_flux_absolute_m4_per_s2,
            ),
            ("riemann_flux_relative", self.riemann_flux_relative),
            ("volume_relative", self.volume_relative),
            ("momentum_relative", self.momentum_relative),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ChannelError::input(format!(
                    "{name}={value}: supply a finite nonnegative tolerance"
                )));
            }
        }
        for (name, absolute) in [
            (
                "riemann_water_flux_absolute_m3_per_s",
                self.riemann_water_flux_absolute_m3_per_s,
            ),
            (
                "riemann_momentum_flux_absolute_m4_per_s2",
                self.riemann_momentum_flux_absolute_m4_per_s2,
            ),
        ] {
            if absolute == 0.0 && self.riemann_flux_relative == 0.0 {
                return Err(ChannelError::input(format!(
                    "{name}={absolute}, riemann_flux_relative={}: supply at least one positive flux tolerance",
                    self.riemann_flux_relative
                )));
            }
        }
        if self.cfl > 0.45 || self.min_step > self.max_step || self.root_iterations == 0 {
            return Err(ChannelError::input(format!(
                "cfl={}, min_step={}, max_step={}, root_iterations={}: require cfl <=0.45, min_step <=max_step, and positive root iterations",
                self.cfl, self.min_step, self.max_step, self.root_iterations
            )));
        }
        Ok(())
    }

    pub(super) fn wave_config(&self) -> SolverConfig {
        SolverConfig {
            gravity: self.gravity,
            cfl: self.cfl,
            max_step: self.max_step,
            min_step: self.min_step,
            max_retries: self.max_retries,
            root_absolute_depth: self.root_absolute_depth,
            root_relative_depth: self.root_relative_depth,
            root_iterations: self.root_iterations,
            volume_absolute: self.volume_absolute,
            volume_relative: self.volume_relative,
            ..SolverConfig::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ChannelConfig, ChannelTriangularReconstruction};

    #[test]
    fn flux_tolerances_require_finite_nonnegative_values_and_positive_pairs() {
        let defaults = ChannelConfig::default();
        assert_eq!(
            defaults.triangular_reconstruction,
            ChannelTriangularReconstruction::LimitedLinear
        );
        assert_eq!(defaults.riemann_water_flux_absolute_m3_per_s, 1e-12);
        assert_eq!(defaults.riemann_momentum_flux_absolute_m4_per_s2, 1e-12);
        assert_eq!(defaults.riemann_flux_relative, 1e-12);
        for invalid in [-1.0, f64::INFINITY, f64::NAN] {
            for index in 0..3 {
                let mut config = defaults.clone();
                match index {
                    0 => config.riemann_water_flux_absolute_m3_per_s = invalid,
                    1 => config.riemann_momentum_flux_absolute_m4_per_s2 = invalid,
                    _ => config.riemann_flux_relative = invalid,
                }
                assert!(config.check().is_err());
            }
        }
        let mut config = defaults;
        config.riemann_flux_relative = 0.0;
        assert!(config.check().is_ok());
        config.riemann_water_flux_absolute_m3_per_s = 0.0;
        assert!(config.check().is_err());
        config.riemann_water_flux_absolute_m3_per_s = 1e-12;
        config.riemann_momentum_flux_absolute_m4_per_s2 = 0.0;
        assert!(config.check().is_err());
        config.riemann_flux_relative = 1e-12;
        config.riemann_water_flux_absolute_m3_per_s = 0.0;
        assert!(config.check().is_ok());
    }
}
