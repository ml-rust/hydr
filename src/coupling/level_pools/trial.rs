use super::forcing::BoundPoolForcing;
use super::inflows::{BoundInflows, InflowState, InflowStep};
use super::mass::InflowMass;
use super::native_accounting::PoolCouplingInterval;
use super::{
    CoupledLevelPoolBatch, CoupledLevelPoolError, CoupledLevelPoolForcingInterval,
    CoupledLevelPoolStepDiagnostics, exchange::Exchange, heads::Heads,
};
use crate::storage::{LevelPoolBatch, LevelPoolError, LevelPoolSnapshot, LevelPoolSourcePlan};
use crate::structures::{StructureError, StructureInputErrorKind};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::AndersonAlgorithms;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(super) struct StoredStart<R: Runtime<DType = DType>> {
    pub pools: LevelPoolSnapshot<R>,
    pub external: InflowState<R>,
}

#[derive(Clone, Debug)]
pub(super) struct StoredStep<R: Runtime<DType = DType>> {
    pub start_frame: StoredStart<R>,
    pub external: InflowStep<R>,
    pub forcing_interval: Option<BoundPoolForcing<R>>,
    pub coupling: PoolCouplingInterval,
    pub input: Tensor<R>,
    pub exchange: Exchange<R>,
    pub heads: Heads<R>,
    pub start: f64,
    pub end: f64,
    pub requested: f64,
    pub iterations: usize,
    pub retries: usize,
    pub head_residual: f64,
    pub transfer_residual: f64,
    pub mass_residual: f64,
}
/// Complete immutable candidate. Dropping it leaves accepted storage unchanged.
#[derive(Debug)]
pub struct CoupledLevelPoolTrial<R: Runtime<DType = DType>> {
    pub(super) next: CoupledLevelPoolBatch<R>,
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
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>,
{
    /// Solve one bounded interval, returning its actual endpoint after any shortening.
    pub fn trial_to(
        &self,
        requested_end: f64,
    ) -> Result<CoupledLevelPoolTrial<R>, CoupledLevelPoolError> {
        if self.source_policies().is_some() || !self.setup().external_inflows.is_empty() {
            return Err(CoupledLevelPoolError::input(
                "forced setup lacks interval inputs. Supply trial_with_forcing_to with complete coverage",
            ));
        }
        let inflows = self
            .context
            .inflows
            .bind_inputs(&self.context.client, &[])?;
        self.trial_interval(requested_end, None, None, &inflows)
    }
    /// Solve covered local sources and external inflows, preparing each attempted duration independently.
    pub fn trial_with_forcing_to(
        &self,
        requested_end: f64,
        interval: &CoupledLevelPoolForcingInterval,
    ) -> Result<CoupledLevelPoolTrial<R>, CoupledLevelPoolError> {
        interval.check(
            self.time(),
            requested_end,
            self.context.pool_count(),
            self.source_policies().is_some(),
        )?;
        let plan = interval
            .sources
            .as_ref()
            .map(|sources| self.pools.bind_source_rates(sources))
            .transpose()?;
        let forcing = BoundPoolForcing {
            start_s: interval.start_s,
            end_s: interval.end_s,
            sources: interval.sources.clone(),
            inflows: self
                .context
                .inflows
                .bind_inputs(&self.context.client, &interval.inflows)?,
        };
        self.trial_interval(
            requested_end,
            Some(&forcing),
            plan.as_ref(),
            &forcing.inflows,
        )
    }

    pub(crate) fn trial_with_native_inflows_to(
        &self,
        requested_end: f64,
        sources: Option<&[crate::storage::LevelPoolSources]>,
        volumes: &Tensor<R>,
    ) -> Result<CoupledLevelPoolTrial<R>, CoupledLevelPoolError> {
        super::forcing::check_coverage(
            (self.time(), requested_end),
            (self.time(), requested_end),
            sources.map(|sources| sources.len()),
            self.context.pool_count(),
            self.source_policies().is_some(),
            true,
        )?;
        let plan = sources
            .map(|sources| self.pools.bind_source_rates(sources))
            .transpose()?;
        let forcing = BoundPoolForcing {
            start_s: self.time(),
            end_s: requested_end,
            sources: sources.map(|sources| sources.to_vec()),
            inflows: self
                .context
                .inflows
                .bind_interval_volumes(&self.context.client, volumes)?,
        };
        self.trial_interval(
            requested_end,
            Some(&forcing),
            plan.as_ref(),
            &forcing.inflows,
        )
    }
    fn trial_interval(
        &self,
        requested_end: f64,
        forcing_interval: Option<&BoundPoolForcing<R>>,
        source_plan: Option<&LevelPoolSourcePlan<'_, R>>,
        inflows: &BoundInflows<R>,
    ) -> Result<CoupledLevelPoolTrial<R>, CoupledLevelPoolError> {
        let start = self.time();
        if let Some(forcing) = forcing_interval {
            forcing.check(
                start,
                requested_end,
                self.context.pool_count(),
                self.source_policies().is_some(),
            )?;
        }
        let config = &self.context.setup.config;
        if !requested_end.is_finite() || requested_end <= start {
            return Err(CoupledLevelPoolError::input(format!(
                "requested end={requested_end}, accepted time={start}: supply a finite later endpoint"
            )));
        }
        let mut end = requested_end.min(start + config.maximum_step_s);
        if inflows.has_interval_volumes() && end < requested_end {
            return Err(recompute_request(
                start,
                requested_end,
                end,
                config.minimum_step_s,
                "requested duration exceeds the configured maximum step".into(),
            ));
        }
        let start_heads = self.context.heads(
            &self.pools.stages_tensor()?,
            &self.context.wet(self.pools.volumes_tensor())?,
        )?;
        self.context.structures.evaluate_heads(
            &start_heads.a,
            &start_heads.b,
            &self.context.bed_a,
            &self.context.bed_b,
            &start_heads.wet_a,
            &start_heads.wet_b,
        )?;
        let mut retries = 0;
        loop {
            let dt = end - start;
            if !dt.is_finite() || dt < config.minimum_step_s || end <= start {
                return Err(CoupledLevelPoolError::Unresolved {
                    dt_s: dt,
                    retries,
                    reason: "requested interval cannot represent an admitted minimum step".into(),
                });
            }
            match self.attempt(
                end,
                requested_end,
                retries,
                forcing_interval,
                source_plan,
                inflows,
            ) {
                Ok(next) => return Ok(CoupledLevelPoolTrial { next }),
                Err(error) => {
                    if !retryable(&error) {
                        return Err(error);
                    }
                    let shorter = start + dt * 0.5;
                    if inflows.has_interval_volumes() {
                        return Err(recompute_request(
                            start,
                            requested_end,
                            shorter,
                            config.minimum_step_s,
                            error.to_string(),
                        ));
                    }
                    if retries >= config.retry_limit
                        || shorter - start < config.minimum_step_s
                        || shorter <= start
                        || shorter >= end
                    {
                        return Err(CoupledLevelPoolError::Unresolved {
                            dt_s: dt,
                            retries,
                            reason: error.to_string(),
                        });
                    }
                    retries += 1;
                    end = shorter;
                }
            }
        }
    }
    fn attempt(
        &self,
        end: f64,
        requested: f64,
        retries: usize,
        forcing_interval: Option<&BoundPoolForcing<R>>,
        source_plan: Option<&LevelPoolSourcePlan<'_, R>>,
        inflows: &BoundInflows<R>,
    ) -> Result<Self, CoupledLevelPoolError> {
        let steps = self.pools.accepted_steps().checked_add(1).ok_or_else(|| {
            CoupledLevelPoolError::numerical(
                "accepted step count overflows. Start a bounded continuation",
            )
        })?;
        let external = self.context.inflows.prepare(
            &self.context.client,
            inflows,
            &self.external,
            end - self.time(),
            steps,
        )?;
        let external_before = self
            .context
            .inflows
            .state_summary(&self.context.client, &self.external)?;
        let external_totals = self
            .context
            .inflows
            .state_summary(&self.context.client, &external.next)?;
        let prepared = source_plan.map(|plan| plan.prepare(end)).transpose()?;
        let sources = prepared.as_ref().map(|p| p.source_totals()).transpose()?;
        let super::iteration::SolvedInterval {
            input,
            evaluation,
            iterations,
            head_residual,
            transfer_residual,
        } = super::iteration::solve(
            &self.context,
            &self.pools,
            end,
            prepared.as_ref(),
            &external,
        )?;
        let checked_mass = self.context.mass_check(
            &self.pools,
            &evaluation.pools,
            evaluation.exchange.aggregation_bound,
            sources.as_ref(),
            InflowMass {
                before: &external_before,
                interval: &external.summary,
                cumulative: &external_totals,
            },
        )?;
        let mass_residual = checked_mass.residual;
        let coupling = self.context.interval_summary(
            &checked_mass.before,
            &checked_mass.after,
            (self.time(), end),
            evaluation.exchange.aggregation_bound,
            sources.as_ref(),
            InflowMass {
                before: &external_before,
                interval: &external.summary,
                cumulative: &external_totals,
            },
        )?;
        let step = StoredStep {
            coupling,
            start_frame: StoredStart {
                pools: self.pools.snapshot(),
                external: self.external.clone(),
            },
            external: external.step,
            forcing_interval: forcing_interval.cloned(),
            input,
            exchange: evaluation.exchange,
            heads: evaluation.heads,
            start: self.time(),
            end,
            requested,
            iterations,
            retries,
            head_residual,
            transfer_residual,
            mass_residual,
        };
        Ok(Self {
            context: Arc::clone(&self.context),
            pools: evaluation.pools,
            external: external.next,
            last: Some(Arc::new(step)),
        })
    }
}
impl<R: Runtime<DType = DType>> CoupledLevelPoolTrial<R>
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
    /// Accept all candidate storage and receipts by ownership.
    pub fn into_next(self) -> CoupledLevelPoolBatch<R> {
        self.next
    }
    /// Accepted candidate time in seconds.
    pub fn time(&self) -> f64 {
        self.next.time()
    }
    /// Candidate pool storage and accounting.
    pub fn pools(&self) -> &LevelPoolBatch<R> {
        self.next.pools()
    }
    /// Explicitly project complete candidate transfer receipts.
    pub fn diagnostics(&self) -> Result<CoupledLevelPoolStepDiagnostics, CoupledLevelPoolError> {
        self.next.last_step()?.ok_or_else(|| {
            CoupledLevelPoolError::numerical("candidate lacks coupled interval receipts")
        })
    }
}
fn retryable(error: &CoupledLevelPoolError) -> bool {
    matches!(
        error,
        CoupledLevelPoolError::Numerical(_)
            | CoupledLevelPoolError::Pool(LevelPoolError::Numerical { .. })
            | CoupledLevelPoolError::Structure(StructureError::Numerical { .. })
            | CoupledLevelPoolError::Structure(StructureError::InvalidInput {
                kind: StructureInputErrorKind::UnsupportedRegime,
                ..
            })
    )
}

fn recompute_request(
    start: f64,
    requested: f64,
    suggested: f64,
    minimum_step: f64,
    reason: String,
) -> CoupledLevelPoolError {
    let dt = suggested - start;
    if !suggested.is_finite()
        || !dt.is_finite()
        || suggested <= start
        || suggested >= requested
        || dt < minimum_step
    {
        CoupledLevelPoolError::Unresolved {
            dt_s: requested - start,
            retries: 0,
            reason,
        }
    } else {
        CoupledLevelPoolError::RecomputeInterval {
            start_s: start,
            requested_end_s: requested,
            suggested_end_s: suggested,
            reason,
        }
    }
}
