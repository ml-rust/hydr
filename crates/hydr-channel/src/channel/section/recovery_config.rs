use super::CrossSectionError;

/// Local-height convergence and independent forward-area admission tolerances.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossSectionRecoveryConfig {
    /// Absolute local-height bracket tolerance, in metres.
    pub height_atol_m: f64,
    /// Relative local-height bracket tolerance.
    pub height_rtol: f64,
    /// Absolute forward-area residual tolerance, in square metres.
    pub area_atol_m2: f64,
    /// Relative forward-area residual tolerance against the requested area.
    pub area_rtol: f64,
    /// Maximum bisection rounds without bracket expansion.
    pub max_iter: usize,
}

impl Default for CrossSectionRecoveryConfig {
    fn default() -> Self {
        Self {
            height_atol_m: 1e-14,
            height_rtol: 1e-14,
            area_atol_m2: 1e-12,
            area_rtol: 1e-12,
            max_iter: 256,
        }
    }
}

impl CrossSectionRecoveryConfig {
    pub(super) fn check(&self) -> Result<(), CrossSectionError> {
        for (name, value) in [
            ("height_atol_m", self.height_atol_m),
            ("height_rtol", self.height_rtol),
            ("area_atol_m2", self.area_atol_m2),
            ("area_rtol", self.area_rtol),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(CrossSectionError::input(format!(
                    "{name}={value}: supply a finite nonnegative tolerance"
                )));
            }
        }
        for (name, absolute, relative) in [
            ("height", self.height_atol_m, self.height_rtol),
            ("area", self.area_atol_m2, self.area_rtol),
        ] {
            if absolute == 0.0 && relative == 0.0 {
                return Err(CrossSectionError::input(format!(
                    "{name} tolerances are both zero: supply a positive tolerance"
                )));
            }
        }
        if self.max_iter == 0 {
            return Err(CrossSectionError::input(
                "max_iter=0: supply a positive bisection iteration limit",
            ));
        }
        Ok(())
    }
}
