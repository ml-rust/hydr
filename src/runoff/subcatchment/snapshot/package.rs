use super::super::{
    SubcatchmentBatch, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentError,
    SubcatchmentLedger, SubcatchmentState, SubcatchmentStepDiagnostics,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

/// Caller-owned complete storage, soil, and accounting continuation values.
///
/// Preserve every F64 bit during encoding. Import checks consistency, not historical authenticity.
/// Exact continuation requires the same build, runtime, batch layout, and accepted interval partition.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentSnapshotData {
    /// Complete ordered names, physical geometry, soil parameters, and routing.
    pub definitions: Vec<SubcatchmentDefinition>,
    /// Root, water residual, and separate soil adjustment controls.
    pub config: SubcatchmentConfig,
    /// Accepted elapsed time in seconds.
    pub time: f64,
    /// Complete accepted subarea depths and native pervious soil memory.
    pub states: Vec<SubcatchmentState>,
    /// Cumulative external accounting, internal traffic, and numerical uncertainty consumption.
    pub ledger: SubcatchmentLedger,
    /// Number of accepted intervals.
    pub accepted_steps: usize,
    /// Complete final source, transfer, and drainage receipts.
    pub last_step: Option<SubcatchmentStepDiagnostics>,
}

/// Trusted immutable clone of complete numerical continuation state.
#[derive(Clone, Debug)]
pub struct SubcatchmentSnapshot<R: Runtime<DType = DType>> {
    pub(super) batch: SubcatchmentBatch<R>,
}

impl<R> SubcatchmentSnapshot<R>
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
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Project complete values for caller-owned encoding and persistence.
    pub fn data(&self) -> Result<SubcatchmentSnapshotData, SubcatchmentError> {
        self.batch.snapshot_data()
    }

    /// Accepted elapsed time in seconds.
    pub fn time(&self) -> f64 {
        self.batch.time()
    }
}

impl<R> SubcatchmentBatch<R>
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
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Clone complete state without projecting tensors or advancing either soil group.
    pub fn snapshot(&self) -> SubcatchmentSnapshot<R> {
        SubcatchmentSnapshot {
            batch: self.clone(),
        }
    }

    /// Project complete values for caller-owned encoding and persistence.
    pub fn snapshot_data(&self) -> Result<SubcatchmentSnapshotData, SubcatchmentError> {
        Ok(SubcatchmentSnapshotData {
            definitions: self.definitions().to_vec(),
            config: self.config().clone(),
            time: self.time(),
            states: self.states()?,
            ledger: self.ledger().clone(),
            accepted_steps: self.accepted_steps(),
            last_step: self.last_step()?,
        })
    }
}
