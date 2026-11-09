//! Checked native parameters with explicit SI conversion.

use super::SoilError;

pub(crate) const METRES_PER_FOOT: f64 = 0.3048;

/// Immutable parameters for the pinned EPA Modified Green–Ampt model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModifiedGreenAmptParameters {
    pub(crate) suction: f64,
    pub(crate) conductivity: f64,
    pub(crate) deficit: f64,
    pub(crate) lu: f64,
    pub(crate) capacity: f64,
    pub(crate) deficit_ceiling: f64,
}

impl ModifiedGreenAmptParameters {
    /// Check suction in feet, conductivity in feet/second, and maximum moisture deficit.
    pub fn from_native(
        suction_ft: f64,
        conductivity_ft_per_s: f64,
        maximum_deficit: f64,
    ) -> Result<Self, SoilError> {
        if !suction_ft.is_finite()
            || suction_ft < 0.0
            || !conductivity_ft_per_s.is_finite()
            || conductivity_ft_per_s <= 0.0
            || !maximum_deficit.is_finite()
            || !(0.0..=1.0).contains(&maximum_deficit)
        {
            return Err(SoilError::input(format!(
                "S={suction_ft}, Ks={conductivity_ft_per_s}, IMDmax={maximum_deficit}: require finite S>=0, Ks>0, and IMDmax in [0,1]"
            )));
        }
        let ksat = conductivity_ft_per_s * 12.0 * 3600.0;
        let lu = 4.0 * ksat.sqrt() / 12.0;
        let capacity = maximum_deficit * lu;
        let si_ks = conductivity_ft_per_s * METRES_PER_FOOT;
        let si_s = suction_ft * METRES_PER_FOOT;
        if !lu.is_finite()
            || lu <= 0.0
            || !capacity.is_finite()
            || (maximum_deficit > 0.0 && capacity == 0.0)
            || !si_ks.is_finite()
            || si_ks <= 0.0
            || !si_s.is_finite()
            || (suction_ft != 0.0 && si_s == 0.0)
            || !(5400.0 / lu).is_finite()
        {
            return Err(SoilError::input(format!(
                "S_ft={suction_ft}, Ks_ft_per_s={conductivity_ft_per_s}, IMDmax={maximum_deficit}, Lu_ft={lu}, capacity_ft={capacity}, recovery_time_s={}, S_m={si_s}, Ks_m_per_s={si_ks}: derived parameter or unit bridge is unrepresentable. Rescale parameters",
                5400.0 / lu
            )));
        }
        Ok(Self {
            suction: suction_ft,
            conductivity: conductivity_ft_per_s,
            deficit: maximum_deficit,
            lu,
            capacity,
            deficit_ceiling: maximum_deficit.max(capacity / lu),
        })
    }

    /// Check suction in metres, conductivity in metres/second, and maximum moisture deficit.
    pub fn from_si(
        suction_m: f64,
        conductivity_m_per_s: f64,
        maximum_deficit: f64,
    ) -> Result<Self, SoilError> {
        if !suction_m.is_finite() || !conductivity_m_per_s.is_finite() {
            return Err(SoilError::input(format!(
                "S_m={suction_m}, Ks_m_per_s={conductivity_m_per_s}, IMDmax={maximum_deficit}: supply finite SI suction and conductivity"
            )));
        }
        Self::from_native(
            suction_m / METRES_PER_FOOT,
            conductivity_m_per_s / METRES_PER_FOOT,
            maximum_deficit,
        )
        .map_err(|error| {
            SoilError::input(format!(
                "SI bridge S_m={suction_m}, Ks_m_per_s={conductivity_m_per_s}, IMDmax={maximum_deficit}: {error}"
            ))
        })
    }

    /// Native suction in feet.
    pub fn suction_ft(self) -> f64 {
        self.suction
    }
    /// Native saturated conductivity in feet/second.
    pub fn conductivity_ft_per_s(self) -> f64 {
        self.conductivity
    }
    /// Maximum moisture deficit.
    pub fn maximum_deficit(self) -> f64 {
        self.deficit
    }
    /// Original native upper-zone depth in feet.
    pub fn upper_zone_depth_ft(self) -> f64 {
        self.lu
    }
    /// Native upper-zone capacity in feet.
    pub fn capacity_ft(self) -> f64 {
        self.capacity
    }
    /// Maximum admitted deficit, including the exact recovery arithmetic rounding band.
    pub fn recovered_deficit_ceiling(self) -> f64 {
        self.deficit_ceiling
    }
}
