use super::CoupledLevelPoolError;
use crate::storage::LevelPoolSources;

/// Complete constant local source rates over an explicit coverage interval.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolSourceInterval {
    /// Earliest covered time in seconds.
    pub start_s: f64,
    /// Latest covered time in seconds, including the original requested endpoint.
    pub end_s: f64,
    /// One checked rain, bottom-loss, and evaporation record per ordered pool.
    pub sources: Vec<LevelPoolSources>,
}

impl CoupledLevelPoolSourceInterval {
    pub(super) fn check(
        &self,
        accepted: f64,
        requested: f64,
        count: usize,
        enabled: bool,
    ) -> Result<(), CoupledLevelPoolError> {
        if !enabled {
            return Err(CoupledLevelPoolError::input(
                "source interval supplied to a batch without source policies. Construct with new_with_sources",
            ));
        }
        if !self.start_s.is_finite()
            || !self.end_s.is_finite()
            || !requested.is_finite()
            || self.start_s > accepted
            || accepted >= requested
            || requested > self.end_s
            || self.sources.len() != count
        {
            return Err(CoupledLevelPoolError::input(format!(
                "source interval=[{},{}], accepted={accepted}, requested={requested}, records={}, pools={count}: supply complete finite coverage and one original source record per pool",
                self.start_s,
                self.end_s,
                self.sources.len()
            )));
        }
        Ok(())
    }
    pub(super) fn same_bits(&self, other: &Self) -> bool {
        self.start_s.to_bits() == other.start_s.to_bits()
            && self.end_s.to_bits() == other.end_s.to_bits()
            && self.sources.len() == other.sources.len()
            && self.sources.iter().zip(&other.sources).all(|(a, b)| {
                crate::structures::identity::bits(
                    &[
                        a.rain_rate_m_per_s,
                        a.potential_evaporation_rate_m_per_s,
                        a.bottom_loss_rate_m3_per_s,
                    ],
                    &[
                        b.rain_rate_m_per_s,
                        b.potential_evaporation_rate_m_per_s,
                        b.bottom_loss_rate_m3_per_s,
                    ],
                )
            })
    }
}
