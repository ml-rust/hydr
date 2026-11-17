use super::super::{
    LevelPoolBatch, LevelPoolConfig, LevelPoolElement, LevelPoolElementLedger, LevelPoolError,
    LevelPoolState, LevelPoolStepDiagnostics,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;

/// Schema identity of the level-pool snapshot package.
pub const LEVEL_POOL_SNAPSHOT_SCHEMA: &str = "hydr-level-pool-checkpoint-1";
/// Numerical contract identity of the level-pool snapshot package.
pub const LEVEL_POOL_CONTRACT_VERSION: &str = "hydr-level-pool-numr-cpu-1";

/// Complete caller-owned values. Preserve every F64 bit when encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolSnapshotData {
    /// Package schema identity.
    pub schema: String,
    /// Numerical contract identity.
    pub contract: String,
    /// Complete immutable source-enabling evidence, absent for exchange-only state.
    pub source_policies: Option<Vec<crate::storage::LevelPoolSourcePolicy>>,
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
pub struct LevelPoolSnapshot<R: Runtime<DType = DType>> {
    pub(super) batch: LevelPoolBatch<R>,
}
impl<R> LevelPoolSnapshot<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    /// Project complete values for caller-owned encoding.
    pub fn data(&self) -> Result<LevelPoolSnapshotData, LevelPoolError> {
        self.batch.snapshot_data()
    }
    /// Accepted elapsed time in seconds.
    pub fn time(&self) -> f64 {
        self.batch.time()
    }
}
impl<R> LevelPoolBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    /// Clone complete persistent continuation state without numerical projection.
    pub fn snapshot(&self) -> LevelPoolSnapshot<R> {
        LevelPoolSnapshot {
            batch: self.clone(),
        }
    }
    /// Project complete continuation values on explicit request.
    pub fn snapshot_data(&self) -> Result<LevelPoolSnapshotData, LevelPoolError> {
        Ok(LevelPoolSnapshotData {
            schema: LEVEL_POOL_SNAPSHOT_SCHEMA.into(),
            contract: LEVEL_POOL_CONTRACT_VERSION.into(),
            source_policies: self.source_policies().map(<[_]>::to_vec),
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
