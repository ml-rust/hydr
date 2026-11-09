//! Independent root-width and physical volume admission controls.

use super::ReservoirError;

/// Numerical work and residual controls for the standalone reservoir.
#[derive(Clone, Debug, PartialEq)]
pub struct ReservoirConfig {
    /// Absolute excess-root bracket width in metres.
    pub root_absolute_depth: f64,
    /// Relative excess-root bracket width.
    pub root_relative_depth: f64,
    /// Maximum batched bisection rounds, without bracket expansion.
    pub root_iterations: usize,
    /// Absolute constitutive and water residual tolerance in m³.
    pub volume_absolute: f64,
    /// Relative constitutive and water residual tolerance, without a unit floor.
    pub volume_relative: f64,
}
impl Default for ReservoirConfig {
    fn default() -> Self {
        Self {
            root_absolute_depth: f64::from_bits(1),
            root_relative_depth: 64.0 * f64::EPSILON,
            root_iterations: 512,
            volume_absolute: 1e-9,
            volume_relative: 1e-12,
        }
    }
}
impl ReservoirConfig {
    pub(crate) fn check(&self) -> Result<(), ReservoirError> {
        for (name, value) in [
            ("root_absolute_depth", self.root_absolute_depth),
            ("root_relative_depth", self.root_relative_depth),
            ("volume_absolute", self.volume_absolute),
            ("volume_relative", self.volume_relative),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ReservoirError::input(format!(
                    "{name}={value}: require a finite nonnegative tolerance"
                )));
            }
        }
        if self.root_iterations == 0
            || (self.root_absolute_depth == 0.0 && self.root_relative_depth == 0.0)
            || (self.volume_absolute == 0.0 && self.volume_relative == 0.0)
        {
            return Err(ReservoirError::input(format!(
                "config={self:?}: require positive root work and at least one positive tolerance in each pair"
            )));
        }
        Ok(())
    }
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.root_iterations == other.root_iterations
            && [
                self.root_absolute_depth,
                self.root_relative_depth,
                self.volume_absolute,
                self.volume_relative,
            ]
            .iter()
            .zip([
                other.root_absolute_depth,
                other.root_relative_depth,
                other.volume_absolute,
                other.volume_relative,
            ])
            .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}
