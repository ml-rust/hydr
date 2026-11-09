use super::super::{
    LevelPoolBatch, LevelPoolConfig, LevelPoolElement, LevelPoolElementLedger, LevelPoolError,
    LevelPoolState, LevelPoolStepDiagnostics,
};

/// Complete caller-owned values. Preserve every F64 bit when encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolSnapshotData {
    /// Immutable ordered curves, physical assertions, and identities.
    pub elements: Vec<LevelPoolElement>,
    /// Exact dimensional controls.
    pub config: LevelPoolConfig,
    /// Accepted elapsed time in seconds.
    pub time: f64,
    /// Authoritative current volume per element.
    pub states: Vec<LevelPoolState>,
    /// Original/current volumes, lifetime counters, and conservative diagnostic bounds.
    pub element_ledgers: Vec<LevelPoolElementLedger>,
    /// Number of accepted original intervals.
    pub accepted_steps: usize,
    /// Complete final interval receipts, absent before any accepted exchange.
    pub last_step: Option<LevelPoolStepDiagnostics>,
}

/// Trusted complete persistent state clone without projection.
#[derive(Clone, Debug)]
pub struct LevelPoolSnapshot {
    pub(super) batch: LevelPoolBatch,
}
impl LevelPoolSnapshot {
    /// Project complete values for caller-owned encoding.
    pub fn data(&self) -> Result<LevelPoolSnapshotData, LevelPoolError> {
        self.batch.snapshot_data()
    }
    /// Accepted elapsed time in seconds.
    pub fn time(&self) -> f64 {
        self.batch.time()
    }
}
impl LevelPoolBatch {
    /// Clone complete persistent continuation state without numerical projection.
    pub fn snapshot(&self) -> LevelPoolSnapshot {
        LevelPoolSnapshot {
            batch: self.clone(),
        }
    }
    /// Project complete continuation values on explicit request.
    pub fn snapshot_data(&self) -> Result<LevelPoolSnapshotData, LevelPoolError> {
        Ok(LevelPoolSnapshotData {
            elements: self.elements().to_vec(),
            config: self.config().clone(),
            time: self.time(),
            states: self.states()?,
            element_ledgers: self.element_ledgers()?,
            accepted_steps: self.accepted_steps(),
            last_step: self.last_step()?,
        })
    }
}
