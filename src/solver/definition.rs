//! Immutable numerical definition with its folded forcing history.

use super::forcing_history::{self, ForcingPrefix};
use super::{BoundaryConditions, EngineError, ForcingSchedule, SolverConfig};
use crate::model::Domain;
use crate::soil::LandModel;

/// Engine definition. `forcing` holds the retained window that follows `prefix`.
///
/// `chain[k]` is the prefix digest after folding retained intervals `0..=k`.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Definition {
    pub domain: Domain,
    pub land: LandModel,
    pub boundaries: BoundaryConditions,
    pub forcing: ForcingSchedule,
    pub config: SolverConfig,
    pub prefix: ForcingPrefix,
    pub chain: Vec<[u8; 32]>,
}

impl Definition {
    /// Builds a definition and caches the forcing digest chain once.
    ///
    /// # Errors
    /// Rejects a prefix that does not end at the first retained interval start.
    pub(super) fn new(
        domain: Domain,
        land: LandModel,
        boundaries: BoundaryConditions,
        forcing: ForcingSchedule,
        config: SolverConfig,
        prefix: ForcingPrefix,
    ) -> Result<Self, EngineError> {
        let first = forcing.intervals().first().map(|interval| interval.start);
        if !first.is_some_and(|first| prefix.consistent(first)) {
            return Err(EngineError::rejected(
                "forcing prefix",
                format!(
                    "prefix={prefix:?}, first retained start={first:?}: supply a prefix that ends at the first retained interval start"
                ),
            ));
        }
        let chain = forcing_history::chain(&prefix.digest, forcing.intervals());
        Ok(Self {
            domain,
            land,
            boundaries,
            forcing,
            config,
            prefix,
            chain,
        })
    }

    /// Folds the first `count` retained intervals onto the prefix from the cached chain.
    pub(super) fn folded(&self, count: usize) -> Option<ForcingPrefix> {
        forcing_history::folded(
            &self.prefix,
            self.forcing.intervals(),
            Some(&self.chain),
            count,
        )
    }

    /// Returns the local index of the first interval a checkpoint retains.
    ///
    /// The window starts at the interval containing the last step start.
    /// Without a last step it starts at the interval containing `time`, or the final interval.
    pub(super) fn retained_from(&self, time: f64, last_start: Option<f64>) -> usize {
        let intervals = self.forcing.intervals();
        let at = last_start.unwrap_or(time);
        intervals
            .partition_point(|interval| interval.end <= at)
            .min(intervals.len().saturating_sub(1))
    }
}
