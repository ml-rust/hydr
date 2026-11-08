use super::EngineError;

/// Dimensional numerical controls, independent of physical flood thresholds.
#[derive(Clone, Debug, PartialEq)]
pub struct SolverConfig {
    /// Gravity in m/s².
    pub gravity: f64,
    /// Combined directional Courant ceiling, at most 0.45.
    pub cfl: f64,
    /// Maximum accepted interval in seconds.
    pub max_step: f64,
    /// Minimum numerical interval in seconds, excluding an exact shorter endpoint.
    pub min_step: f64,
    /// Maximum recalculations of a rejected trial.
    pub max_retries: usize,
    /// Absolute Riemann root-bracket width in metres.
    pub root_absolute_depth: f64,
    /// Relative Riemann root-bracket width.
    pub root_relative_depth: f64,
    /// Maximum bracket expansions and bisections per Riemann solve.
    pub root_iterations: usize,
    /// Absolute water residual tolerance in m³.
    pub volume_absolute: f64,
    /// Relative water residual tolerance.
    pub volume_relative: f64,
    /// Momentum resets below this depth in metres. Positive water remains stored.
    pub momentum_film_depth: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            cfl: 0.45,
            max_step: 1.0,
            min_step: 1e-12,
            max_retries: 64,
            root_absolute_depth: 1e-12,
            root_relative_depth: 1e-12,
            root_iterations: 128,
            volume_absolute: 1e-9,
            volume_relative: 1e-12,
            momentum_film_depth: 0.0,
        }
    }
}

impl SolverConfig {
    pub(super) fn check(&self) -> Result<(), EngineError> {
        for (field, value) in [
            ("gravity", self.gravity),
            ("cfl", self.cfl),
            ("max_step", self.max_step),
            ("min_step", self.min_step),
            ("root_absolute_depth", self.root_absolute_depth),
            ("root_relative_depth", self.root_relative_depth),
            ("volume_absolute", self.volume_absolute),
            ("volume_relative", self.volume_relative),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(EngineError::rejected(
                    "config",
                    format!("{field}={value}: supply a finite positive value"),
                ));
            }
        }
        if self.cfl > 0.45
            || self.min_step > self.max_step
            || self.max_retries == 0
            || self.root_iterations == 0
            || !self.momentum_film_depth.is_finite()
            || self.momentum_film_depth < 0.0
        {
            return Err(EngineError::rejected(
                "config",
                format!(
                    "{self:?}: require CFL <=0.45, min_step <=max_step, positive iteration limits, and a finite nonnegative film depth"
                ),
            ));
        }
        Ok(())
    }
}
