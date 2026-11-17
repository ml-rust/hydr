use super::super::{
    ReservoirBatch, ReservoirConfig, ReservoirElement, ReservoirError, ReservoirLedger,
    ReservoirState, ReservoirStepDiagnostics,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

/// Schema identity of the reservoir snapshot package.
pub const RESERVOIR_SNAPSHOT_SCHEMA: &str = "hydr-reservoir-checkpoint-1";
/// Numerical contract identity of the reservoir snapshot package.
pub const RESERVOIR_CONTRACT_VERSION: &str = "hydr-reservoir-numr-cpu-1";

/// Caller-owned complete reservoir continuation values.
///
/// Preserve every F64 bit when encoding. Checks establish consistency, not historical authenticity.
/// Exact continuation requires the same build, runtime, batch layout, and accepted interval partition.
#[derive(Clone, Debug, PartialEq)]
pub struct ReservoirSnapshotData {
    /// Package schema identity.
    pub schema: String,
    /// Numerical contract identity.
    pub contract: String,
    /// Ordered named elements and their complete checked physical parameters.
    pub elements: Vec<ReservoirElement>,
    /// Numerical controls and independent volume tolerances.
    pub config: ReservoirConfig,
    /// Accepted elapsed time in seconds.
    pub time: f64,
    /// Complete accepted depths in element order.
    pub states: Vec<ReservoirState>,
    /// Independent cumulative storage and transfer accounting.
    pub ledger: ReservoirLedger,
    /// Number of accepted intervals.
    pub accepted_steps: usize,
    /// Complete final accepted interval, absent before the first acceptance.
    pub last_step: Option<ReservoirStepDiagnostics>,
}

/// Trusted immutable clone of complete reservoir continuation state.
#[derive(Clone, Debug)]
pub struct ReservoirSnapshot<R: Runtime<DType = DType>> {
    pub(super) batch: ReservoirBatch<R>,
}

impl<R> ReservoirSnapshot<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Project complete values for caller-owned encoding.
    pub fn data(&self) -> Result<ReservoirSnapshotData, ReservoirError> {
        self.batch.snapshot_data()
    }

    /// Accepted elapsed time in seconds.
    pub fn time(&self) -> f64 {
        self.batch.time()
    }
}

impl<R> ReservoirBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Clone complete numerical state without projecting tensors.
    pub fn snapshot(&self) -> ReservoirSnapshot<R> {
        ReservoirSnapshot {
            batch: self.clone(),
        }
    }

    /// Project complete values for caller-owned encoding and persistence.
    pub fn snapshot_data(&self) -> Result<ReservoirSnapshotData, ReservoirError> {
        Ok(ReservoirSnapshotData {
            schema: RESERVOIR_SNAPSHOT_SCHEMA.into(),
            contract: RESERVOIR_CONTRACT_VERSION.into(),
            elements: self.elements().to_vec(),
            config: self.config().clone(),
            time: self.time(),
            states: self.states()?,
            ledger: self.ledger().clone(),
            accepted_steps: self.accepted_steps(),
            last_step: self.last_step()?,
        })
    }
}
