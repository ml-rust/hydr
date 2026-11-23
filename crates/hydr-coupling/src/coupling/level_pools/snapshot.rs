use super::identity::same_setup;
use super::{
    CoupledLevelPoolBatch, CoupledLevelPoolError, CoupledLevelPoolSetup,
    CoupledLevelPoolStepDiagnostics, trial::StoredStart,
};
use crate::coupling::checkpoint;
use crate::storage::LevelPoolSnapshotData;
use hydr_core::contract::package_identity;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, IndexingOps,
    MidpointOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Device;
use numr::runtime::{Runtime, RuntimeClient};
use solvr::interpolate::Interp1dAlgorithms;
use std::sync::Arc;

/// Schema identity of the coupled level pool snapshot package.
pub const COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA: &str = "hydr-coupled-level-pool-checkpoint-3";
/// Numerical contract identity of the coupled level pool snapshot package.
pub const COUPLED_LEVEL_POOL_CONTRACT_VERSION: &str = "hydr-coupled-level-pool-numr-cpu-2";

/// One native pool and external ledger frame, without recursive coupled history.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolStartData {
    /// Complete native storage and source counters at the retained start.
    pub pools: LevelPoolSnapshotData,
    /// Complete cumulative external volume and bounds at the same start.
    pub external_inflows: Vec<super::CoupledPoolInflowLedger>,
}

/// Complete caller-owned continuation. Preserve every F64 bit when encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolSnapshotData {
    /// Package schema identity.
    pub schema: String,
    /// Numerical contract identity.
    pub contract: String,
    /// Original immutable physical definitions, operations, and numerical budgets.
    pub setup: CoupledLevelPoolSetup,
    /// Complete accepted native storage and cumulative accounting.
    pub pools: LevelPoolSnapshotData,
    /// Complete final coupled interval, absent before any accepted step.
    pub last_step: Option<CoupledLevelPoolStepDiagnostics>,
    /// One bounded native start frame, without coupled history.
    pub start: Option<CoupledLevelPoolStartData>,
    /// Complete cumulative external ledger in immutable boundary order.
    pub external_inflows: Vec<super::CoupledPoolInflowLedger>,
}
impl CoupledLevelPoolSnapshotData {
    /// Check this package identity, then the identity of every nested level-pool package.
    pub(crate) fn check_identities(&self) -> Result<(), String> {
        package_identity(
            "coupled level pool",
            &self.schema,
            COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA,
            &self.contract,
            COUPLED_LEVEL_POOL_CONTRACT_VERSION,
        )?;
        checkpoint::level_pool("coupled level pool checkpoint pools", &self.pools)?;
        if let Some(start) = &self.start {
            checkpoint::level_pool("coupled level pool checkpoint start.pools", &start.pools)?;
        }
        Ok(())
    }
}
/// Trusted device-resident continuation without a host projection.
#[derive(Clone, Debug)]
pub struct CoupledLevelPoolSnapshot<R: Runtime<DType = DType>> {
    batch: CoupledLevelPoolBatch<R>,
}
impl<R: Runtime<DType = DType>> CoupledLevelPoolSnapshot<R>
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
        + Interp1dAlgorithms<R>,
{
    /// Explicitly read complete values for caller-owned encoding.
    pub fn data(&self) -> Result<CoupledLevelPoolSnapshotData, CoupledLevelPoolError> {
        self.batch.snapshot_data()
    }
    /// Accepted time in seconds.
    pub fn time(&self) -> f64 {
        self.batch.time()
    }
}
impl<R: Runtime<DType = DType>> CoupledLevelPoolBatch<R>
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
        + Interp1dAlgorithms<R>,
{
    /// Clone complete trusted state without reading numerical fields.
    pub fn snapshot(&self) -> CoupledLevelPoolSnapshot<R> {
        CoupledLevelPoolSnapshot {
            batch: self.clone(),
        }
    }
    /// Explicitly read complete state and its single bounded start frame.
    pub fn snapshot_data(&self) -> Result<CoupledLevelPoolSnapshotData, CoupledLevelPoolError> {
        Ok(CoupledLevelPoolSnapshotData {
            schema: COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA.into(),
            contract: COUPLED_LEVEL_POOL_CONTRACT_VERSION.into(),
            setup: self.setup().clone(),
            pools: self.pools.snapshot_data()?,
            last_step: self.last_step()?,
            external_inflows: self.external_inflows()?,
            start: self
                .last
                .as_ref()
                .map(|step| -> Result<_, CoupledLevelPoolError> {
                    Ok(CoupledLevelPoolStartData {
                        pools: step.start_frame.pools.data()?,
                        external_inflows: self
                            .context
                            .inflows
                            .project_state(&self.context.client, &step.start_frame.external)?,
                    })
                })
                .transpose()?,
        })
    }
    /// Restore trusted state while retaining the destination runtime clients.
    pub fn restore(
        &mut self,
        snapshot: &CoupledLevelPoolSnapshot<R>,
    ) -> Result<(), CoupledLevelPoolError> {
        require(
            same_setup(self.setup(), snapshot.batch.setup()),
            "immutable setup bits differ from the destination",
        )?;
        require(
            self.context
                .client
                .device()
                .is_same(snapshot.batch.context.client.device()),
            "trusted snapshot device differs from the destination",
        )?;
        let mut pools = self.pools.clone();
        pools.restore(&snapshot.batch.pools.snapshot())?;
        let external = self.context.inflows.restore_state(
            &self.context.client,
            &snapshot.batch.external,
            pools.accepted_steps(),
        )?;
        let last = if let Some(step) = &snapshot.batch.last {
            let mut start = self.pools.clone();
            start.restore(&step.start_frame.pools)?;
            let start_external = self.context.inflows.restore_state(
                &self.context.client,
                &step.start_frame.external,
                start.accepted_steps(),
            )?;
            let mut stored = (**step).clone();
            stored.forcing_interval = stored
                .forcing_interval
                .as_ref()
                .map(|forcing| forcing.restore(&self.context.client, &self.context.inflows))
                .transpose()?;
            stored.start_frame = StoredStart {
                pools: start.snapshot(),
                external: start_external,
            };
            Some(Arc::new(stored))
        } else {
            None
        };
        self.pools = pools;
        self.external = external;
        self.last = last;
        Ok(())
    }
}

pub(super) fn require(condition: bool, message: &str) -> Result<(), CoupledLevelPoolError> {
    if condition {
        Ok(())
    } else {
        Err(CoupledLevelPoolError::input(format!(
            "coupled checkpoint: {message}. Supply an unchanged complete same-build package"
        )))
    }
}
