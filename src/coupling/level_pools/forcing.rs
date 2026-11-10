use super::{CoupledLevelPoolError, CoupledPoolInflowRate};
use crate::storage::LevelPoolSources;

/// Complete constant local source and external inflow rates over an explicit interval.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolForcingInterval {
    /// Earliest covered time in seconds.
    pub start_s: f64,
    /// Latest covered time in seconds, including the original requested endpoint.
    pub end_s: f64,
    /// One checked rain, bottom-loss, and evaporation record per ordered pool.
    pub sources: Option<Vec<LevelPoolSources>>,
    /// Complete original external rates, resolved into immutable binding order during admission.
    pub inflows: Vec<CoupledPoolInflowRate>,
}

impl CoupledLevelPoolForcingInterval {
    pub(super) fn check(
        &self,
        accepted: f64,
        requested: f64,
        count: usize,
        enabled: bool,
    ) -> Result<(), CoupledLevelPoolError> {
        if self.sources.is_some() != enabled {
            return Err(CoupledLevelPoolError::input(
                "forcing source presence differs from setup policies. Supply complete rates exactly when local sources are enabled",
            ));
        }
        if !self.start_s.is_finite()
            || !self.end_s.is_finite()
            || !requested.is_finite()
            || self.start_s > accepted
            || accepted >= requested
            || requested > self.end_s
            || self
                .sources
                .as_ref()
                .is_some_and(|sources| sources.len() != count)
        {
            return Err(CoupledLevelPoolError::input(format!(
                "forcing interval=[{},{}], accepted={accepted}, requested={requested}, source records={:?}, pools={count}: supply finite coverage and complete records for every enabled process",
                self.start_s,
                self.end_s,
                self.sources.as_ref().map(Vec::len)
            )));
        }
        Ok(())
    }
    pub(super) fn same_bits(&self, other: &Self) -> bool {
        let source_bits = match (&self.sources, &other.sources) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.len() == b.len()
                    && a.iter().zip(b).all(|(a, b)| {
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
            _ => false,
        };
        self.start_s.to_bits() == other.start_s.to_bits()
            && self.end_s.to_bits() == other.end_s.to_bits()
            && source_bits
            && self.inflows.len() == other.inflows.len()
            && self.inflows.iter().zip(&other.inflows).all(|(a, b)| {
                a.boundary_id == b.boundary_id
                    && a.input_id == b.input_id
                    && a.rate_m3_per_s.to_bits() == b.rate_m3_per_s.to_bits()
            })
    }
}
