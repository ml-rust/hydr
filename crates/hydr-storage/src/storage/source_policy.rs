//! Explicit immutable source policy and original interval rates.

/// Caller evidence enabling the frozen-area source operator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelPoolSourcePolicy {
    /// Evidence that the first segment represents a rain-receiving basal footprint, including dry storage.
    pub dry_basal_footprint_assertion: String,
    /// Provenance for supplied potential bottom demand, without an inferred soil process.
    pub bottom_loss_provenance: String,
}
impl LevelPoolSourcePolicy {
    pub(crate) fn check(&self) -> Result<(), super::LevelPoolError> {
        super::definition::text(
            &self.dry_basal_footprint_assertion,
            "dry_basal_footprint_assertion",
        )?;
        super::definition::text(&self.bottom_loss_provenance, "bottom_loss_provenance")
    }
}

/// Constant nonnegative source rates over the exact original requested interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelPoolSources {
    /// Rainfall rate in m/s over the frozen accepted-state area.
    pub rain_rate_m_per_s: f64,
    /// Potential evaporation rate in m/s over the same frozen area.
    pub potential_evaporation_rate_m_per_s: f64,
    /// Supplied potential bottom demand in m³/s, capped physically by preceding available water.
    pub bottom_loss_rate_m3_per_s: f64,
}
