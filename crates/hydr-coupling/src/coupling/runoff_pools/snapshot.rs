use super::{RunoffPoolBatch, RunoffPoolError, RunoffPoolSetup, RunoffPoolStepDiagnostics};
use crate::coupling::{CoupledLevelPoolSnapshotData, checkpoint};
use crate::runoff::SubcatchmentSnapshotData;
use hydr_core::contract::package_identity;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, IndexingOps,
    MidpointOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};
use std::sync::Arc;

/// Schema identity of the runoff pool snapshot package.
pub const RUNOFF_POOL_SNAPSHOT_SCHEMA: &str = "hydr-runoff-pool-checkpoint-1";
/// Numerical contract identity of the runoff pool snapshot package.
pub const RUNOFF_POOL_CONTRACT_VERSION: &str = "hydr-runoff-pool-numr-cpu-2";

/// Complete caller-owned paired state, with one bounded common receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct RunoffPoolSnapshotData {
    /// Package schema identity.
    pub schema: String,
    /// Numerical contract identity.
    pub contract: String,
    /// Immutable definitions, mapping, evidence, and numerical controls.
    pub setup: RunoffPoolSetup,
    /// Exact original common epoch, required to match the destination.
    pub initial_time_s: f64,
    /// Complete native runoff, soil, routing, and cumulative accounting.
    pub runoff: SubcatchmentSnapshotData,
    /// Complete native pool, source, and external ledger state.
    pub pools: CoupledLevelPoolSnapshotData,
    /// Final common interval, absent exactly before any accepted interval.
    pub last_step: Option<RunoffPoolStepDiagnostics>,
}
impl RunoffPoolSnapshotData {
    /// Check this package identity, then the identity of every nested package.
    pub(super) fn check_identities(&self) -> Result<(), String> {
        package_identity(
            "runoff pool",
            &self.schema,
            RUNOFF_POOL_SNAPSHOT_SCHEMA,
            &self.contract,
            RUNOFF_POOL_CONTRACT_VERSION,
        )?;
        checkpoint::subcatchment("runoff pool checkpoint runoff", &self.runoff)?;
        self.pools
            .check_identities()
            .map_err(|message| format!("runoff pool checkpoint pools: {message}"))
    }
}
/// Trusted paired tensor continuation without numerical field projection.
#[derive(Clone, Debug)]
pub struct RunoffPoolSnapshot<R: Runtime<DType = DType>> {
    pub(super) batch: RunoffPoolBatch<R>,
}
impl<R: Runtime<DType = DType>> RunoffPoolSnapshot<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Explicitly project complete owned continuation values.
    pub fn data(&self) -> Result<RunoffPoolSnapshotData, RunoffPoolError> {
        self.batch.snapshot_data()
    }
    /// Common accepted time, in seconds.
    pub fn time(&self) -> f64 {
        self.batch.time()
    }
}
impl<R: Runtime<DType = DType>> RunoffPoolBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Clone both trusted child engines and the bounded common receipt without field readback.
    pub fn snapshot(&self) -> RunoffPoolSnapshot<R> {
        RunoffPoolSnapshot {
            batch: self.clone(),
        }
    }
    /// Explicitly project both complete native child packages and the common receipt.
    pub fn snapshot_data(&self) -> Result<RunoffPoolSnapshotData, RunoffPoolError> {
        Ok(RunoffPoolSnapshotData {
            schema: RUNOFF_POOL_SNAPSHOT_SCHEMA.into(),
            contract: RUNOFF_POOL_CONTRACT_VERSION.into(),
            setup: self.setup().clone(),
            initial_time_s: self.initial_time(),
            runoff: self.runoff.snapshot_data()?,
            pools: self.pools.snapshot_data()?,
            last_step: self.last_step()?,
        })
    }
    /// Atomically restore trusted children while retaining the destination configured clients.
    pub fn restore(&mut self, snapshot: &RunoffPoolSnapshot<R>) -> Result<(), RunoffPoolError> {
        self.check_identity(snapshot.batch.setup(), snapshot.batch.initial_time())?;
        let mut runoff = self.runoff.clone();
        runoff.restore(&snapshot.batch.runoff.snapshot())?;
        let mut pools = self.pools.clone();
        pools.restore(&snapshot.batch.pools.snapshot())?;
        let candidate = Self {
            context: Arc::clone(&self.context),
            runoff,
            pools,
            last: snapshot.batch.last.clone(),
        };
        candidate.check_continuation(false)?;
        *self = candidate;
        Ok(())
    }
    pub(super) fn check_identity(
        &self,
        setup: &RunoffPoolSetup,
        epoch: f64,
    ) -> Result<(), RunoffPoolError> {
        if !self.matches_setup(setup) || self.initial_time().to_bits() != epoch.to_bits() {
            return Err(RunoffPoolError::input(
                "composite setup, mapping, or original epoch bits differ from the destination",
            ));
        }
        Ok(())
    }
}
