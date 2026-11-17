use super::forcing::BoundPoolForcing;
use super::identity::{same_external, same_pools, same_setup, same_step};
use super::mass::InflowMass;
use super::snapshot::require;
use super::trial::{StoredStart, StoredStep};
use super::{
    CoupledLevelPoolBatch, CoupledLevelPoolError, CoupledLevelPoolSnapshotData,
    CoupledLevelPoolStepDiagnostics,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use std::sync::Arc;

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
        let external = self.context.inflows.import_state(
            &self.context.client,
            &data.external_inflows,
            pools.accepted_steps(),
        )?;
        let last = match (&data.last_step, &data.start) {
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
                start.restore_data(&start_data.pools)?;
                let start_external = self.context.inflows.import_state(
                    &self.context.client,
                    &start_data.external_inflows,
                    start.accepted_steps(),
                )?;
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
                let source_plan = match &receipt.forcing_interval {
                    Some(interval) => {
                        interval.check(
                            start.time(),
                            receipt.requested_end_s,
                            self.context.pool_count(),
                            start.source_policies().is_some(),
                        )?;
                        interval
                            .sources
                            .as_ref()
                            .map(|sources| start.bind_source_rates(sources))
                            .transpose()?
                    }
                    None => {
                        require(
                            start.source_policies().is_none()
                                && self.setup().external_inflows.is_empty(),
                            "forced setup receipt lacks original forcing",
                        )?;
                        None
                    }
                };
                let original_inputs = receipt
                    .forcing_interval
                    .as_ref()
                    .map_or(&[][..], |forcing| forcing.inflows.as_slice());
                let bound = self
                    .context
                    .inflows
                    .bind_inputs(&self.context.client, original_inputs)?;
                let prepared_external = self.context.inflows.prepare(
                    &self.context.client,
                    &bound,
                    &start_external,
                    receipt.end_s - start.time(),
                    pools.accepted_steps(),
                )?;
                let external_before = self
                    .context
                    .inflows
                    .state_summary(&self.context.client, &start_external)?;
                let external_totals = self
                    .context
                    .inflows
                    .state_summary(&self.context.client, &prepared_external.next)?;
                let prepared = source_plan
                    .as_ref()
                    .map(|plan| plan.prepare(receipt.end_s))
                    .transpose()?;
                let sources = prepared.as_ref().map(|p| p.source_totals()).transpose()?;
                let input = self.context.import_packet(&receipt.iteration)?;
                let evaluated = self.context.evaluate(
                    &start,
                    receipt.end_s,
                    prepared.as_ref(),
                    &prepared_external,
                    &input,
                )?;
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
                let checked_mass = self.context.mass_check(
                    &start,
                    &evaluated.pools,
                    evaluated.exchange.aggregation_bound,
                    sources.as_ref(),
                    InflowMass {
                        before: &external_before,
                        interval: &prepared_external.summary,
                        cumulative: &external_totals,
                    },
                )?;
                require(
                    same_pools(&evaluated.pools.snapshot_data()?, &data.pools),
                    "accepted native storage differs from retained hydraulic operands or cumulative additions",
                )?;
                require(
                    same_external(
                        &self
                            .context
                            .inflows
                            .project_state(&self.context.client, &prepared_external.next)?,
                        &data.external_inflows,
                    ),
                    "external ledger differs from original products or cumulative additions",
                )?;
                let mass_residual = checked_mass.residual;
                let coupling = self.context.interval_summary(
                    &checked_mass.before,
                    &checked_mass.after,
                    (receipt.start_s, receipt.end_s),
                    evaluated.exchange.aggregation_bound,
                    sources.as_ref(),
                    InflowMass {
                        before: &external_before,
                        interval: &prepared_external.summary,
                        cumulative: &external_totals,
                    },
                )?;
                let stored = StoredStep {
                    coupling,
                    start_frame: StoredStart {
                        pools: start.snapshot(),
                        external: start_external,
                    },
                    external: prepared_external.step,
                    forcing_interval: receipt.forcing_interval.as_ref().map(|forcing| {
                        BoundPoolForcing {
                            start_s: forcing.start_s,
                            end_s: forcing.end_s,
                            sources: forcing.sources.clone(),
                            inflows: bound.clone(),
                        }
                    }),
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
        self.external = external;
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
        if step
            .forcing_interval
            .as_ref()
            .is_some_and(|forcing| forcing.requires_exact_span())
        {
            require(
                step.end_s.to_bits() == step.requested_end_s.to_bits() && step.retries == 0,
                "integrated inflow receipt contains a shortened endpoint or child retries",
            )?;
        }
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
