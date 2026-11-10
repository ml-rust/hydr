use super::identity::{same_pools, same_setup, same_step};
use super::{
    CoupledLevelPoolBatch, CoupledLevelPoolError, CoupledLevelPoolSetup,
    CoupledLevelPoolStepDiagnostics, trial::StoredStep,
};
use crate::storage::LevelPoolSnapshotData;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Device;
use numr::runtime::{Runtime, RuntimeClient};
use solvr::interpolate::Interp1dAlgorithms;
use std::sync::Arc;

/// Complete caller-owned continuation. Preserve every F64 bit when encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolSnapshotData {
    /// Original immutable physical definitions, operations, and numerical budgets.
    pub setup: CoupledLevelPoolSetup,
    /// Complete accepted native storage and cumulative accounting.
    pub pools: LevelPoolSnapshotData,
    /// Complete final coupled interval, absent before any accepted step.
    pub last_step: Option<CoupledLevelPoolStepDiagnostics>,
    /// One bounded native start frame, without coupled history.
    pub start_pools: Option<LevelPoolSnapshotData>,
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
            setup: self.setup().clone(),
            pools: self.pools.snapshot_data()?,
            last_step: self.last_step()?,
            start_pools: self
                .last
                .as_ref()
                .map(|s| s.start_pools.data())
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
        let last = if let Some(step) = &snapshot.batch.last {
            let mut start = self.pools.clone();
            start.restore(&step.start_pools)?;
            let mut stored = (**step).clone();
            stored.start_pools = start.snapshot();
            Some(Arc::new(stored))
        } else {
            None
        };
        self.pools = pools;
        self.last = last;
        Ok(())
    }
    /// Check complete owned state and one retained hydraulic map without replaying root iterations.
    pub fn restore_data(
        &mut self,
        data: &CoupledLevelPoolSnapshotData,
    ) -> Result<(), CoupledLevelPoolError> {
        require(
            same_setup(self.setup(), &data.setup),
            "immutable setup bits differ from the destination",
        )?;
        let mut pools = self.pools.clone();
        pools.restore_data(&data.pools)?;
        let last = match (&data.last_step, &data.start_pools) {
            (None, None) => {
                require(
                    pools.accepted_steps() == 0,
                    "advanced state lacks its final coupled interval",
                )?;
                None
            }
            (Some(receipt), Some(start_data)) => {
                require(
                    pools.accepted_steps() > 0,
                    "initial state contains an advanced coupled receipt",
                )?;
                let mut start = self.pools.clone();
                start.restore_data(start_data)?;
                require(
                    start.accepted_steps().checked_add(1) == Some(pools.accepted_steps()),
                    "start and accepted step counts are discontinuous",
                )?;
                require(
                    start.time().to_bits() == receipt.start_s.to_bits()
                        && pools.time().to_bits() == receipt.end_s.to_bits(),
                    "retained interval endpoints differ from native pool times",
                )?;
                self.check_interval(receipt)?;
                let input = self.context.import_packet(&receipt.iteration)?;
                let evaluated = self.context.evaluate(&start, receipt.end_s, &input)?;
                let (head_residual, transfer_residual, flags) =
                    self.context.residuals(&input, &evaluated.packet)?;
                let config = &self.setup().config;
                require(
                    flags
                        && head_residual <= config.head_absolute_m
                        && transfer_residual <= config.transfer_absolute_m3,
                    "retained final iterate exceeds its dimensional budgets or changes support",
                )?;
                self.context.check_endpoint(&evaluated)?;
                let mass_residual = self.context.mass_check(
                    &start,
                    &evaluated.pools,
                    evaluated.exchange.aggregation_bound,
                )?;
                require(
                    same_pools(&evaluated.pools.snapshot_data()?, &data.pools),
                    "accepted native storage differs from retained hydraulic operands or cumulative additions",
                )?;
                let stored = StoredStep {
                    start_pools: start.snapshot(),
                    input,
                    exchange: evaluated.exchange,
                    heads: evaluated.heads,
                    start: receipt.start_s,
                    end: receipt.end_s,
                    requested: receipt.requested_end_s,
                    iterations: receipt.iterations,
                    retries: receipt.retries,
                    head_residual,
                    transfer_residual,
                    mass_residual,
                };
                require(
                    same_step(&self.context.project(&stored)?, receipt),
                    "coupled receipt differs from original laws, donor allocation, dimensional residuals, or physical mass checks",
                )?;
                Some(Arc::new(stored))
            }
            _ => {
                return Err(CoupledLevelPoolError::input(
                    "checkpoint requires both final interval and its native start frame",
                ));
            }
        };
        self.pools = pools;
        self.last = last;
        Ok(())
    }
    fn check_interval(
        &self,
        step: &CoupledLevelPoolStepDiagnostics,
    ) -> Result<(), CoupledLevelPoolError> {
        let c = &self.setup().config;
        require(
            step.start_s.is_finite()
                && step.start_s >= 0.0
                && step.end_s.is_finite()
                && step.requested_end_s.is_finite()
                && step.requested_end_s > step.start_s
                && step.end_s > step.start_s
                && step.end_s <= step.requested_end_s,
            "interval contains invalid original or accepted endpoints",
        )?;
        require(
            step.iterations < c.iteration_limit && step.retries <= c.retry_limit,
            "retained iteration or retry count exceeds original limits",
        )?;
        let mut expected = step.requested_end_s.min(step.start_s + c.maximum_step_s);
        for _ in 0..step.retries {
            let shorter = step.start_s + (expected - step.start_s) * 0.5;
            require(
                shorter > step.start_s && shorter < expected,
                "retained retry sequence stops making progress",
            )?;
            expected = shorter;
        }
        require(
            expected.to_bits() == step.end_s.to_bits()
                && step.end_s - step.start_s >= c.minimum_step_s,
            "accepted endpoint differs from bounded retry arithmetic",
        )
    }
}
fn require(condition: bool, message: &str) -> Result<(), CoupledLevelPoolError> {
    if condition {
        Ok(())
    } else {
        Err(CoupledLevelPoolError::input(format!(
            "coupled checkpoint: {message}. Supply an unchanged complete same-build package"
        )))
    }
}
