use super::ChannelError;
use crate::solver::SolverConfig;

/// Numerical controls for a first-order rectangular channel reach.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelConfig {
    /// Gravity in m/s².
    pub gravity: f64,
    /// Width-aware Courant ceiling, greater than zero and at most 0.45.
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
            gravity: config.gravity,
            cfl: config.cfl,
            max_step: config.max_step,
            min_step: config.min_step,
            max_retries: config.max_retries,
            root_absolute_depth: config.root_absolute_depth,
            root_relative_depth: config.root_relative_depth,
            root_iterations: config.root_iterations,
            volume_absolute: config.volume_absolute,
            volume_relative: config.volume_relative,
            momentum_absolute: 1e-9,
            momentum_relative: 1e-12,
        }
    }
}

impl ChannelConfig {
    pub(super) fn check(&self) -> Result<(), ChannelError> {
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
            ("volume_relative", self.volume_relative),
            ("momentum_relative", self.momentum_relative),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ChannelError::input(format!(
                    "{name}={value}: supply a finite nonnegative tolerance"
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
