//! Independent drainage controls and bounded native-soil reconciliation.

use super::SubcatchmentError;
use crate::runoff::ReservoirConfig;

/// Numerical controls for the complete conservative component.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentConfig {
    /// Existing root, constitutive, and independent water-volume controls.
    pub reservoir: ReservoirConfig,
    /// Lifetime conservative absolute soil water-adjustment budget in m³.
    pub max_cumulative_water_adjustment_m3: f64,
}
impl Default for SubcatchmentConfig {
    fn default() -> Self {
        Self {
            reservoir: ReservoirConfig::default(),
            max_cumulative_water_adjustment_m3: 1e-9,
        }
    }
}
impl SubcatchmentConfig {
    pub(crate) fn check(&self) -> Result<(), SubcatchmentError> {
        self.reservoir.check()?;
        if !self.max_cumulative_water_adjustment_m3.is_finite()
            || self.max_cumulative_water_adjustment_m3 < 0.0
        {
            return Err(SubcatchmentError::input(format!(
                "soil adjustment budget={}: require finite nonnegative m³",
                self.max_cumulative_water_adjustment_m3
            )));
        }
        Ok(())
    }
    #[doc(hidden)]
    pub fn same_bits(&self, other: &Self) -> bool {
        self.reservoir.same_bits(&other.reservoir)
            && self.max_cumulative_water_adjustment_m3.to_bits()
                == other.max_cumulative_water_adjustment_m3.to_bits()
    }
}
