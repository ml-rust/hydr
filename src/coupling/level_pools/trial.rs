use super::{
    CoupledLevelPoolBatch, CoupledLevelPoolError, CoupledLevelPoolSourceInterval,
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
pub(super) struct StoredStep<R: Runtime<DType = DType>> {
    pub start_pools: LevelPoolSnapshot<R>,
    pub source_interval: Option<CoupledLevelPoolSourceInterval>,
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
        if self.source_policies().is_some() {
            return Err(CoupledLevelPoolError::input(
                "source-enabled trial lacks forcing. Supply trial_with_sources_to with complete interval coverage",
            ));
        }
        self.trial_interval(requested_end, None, None)
    }
    /// Solve with complete covered local rates, preparing sources again after each shortening.
    pub fn trial_with_sources_to(
        &self,
        requested_end: f64,
        interval: &CoupledLevelPoolSourceInterval,
    ) -> Result<CoupledLevelPoolTrial<R>, CoupledLevelPoolError> {
        interval.check(
            self.time(),
            requested_end,
            self.context.pool_count(),
            self.source_policies().is_some(),
        )?;
        let plan = self.pools.bind_source_rates(&interval.sources)?;
        self.trial_interval(requested_end, Some(interval), Some(&plan))
    }
    fn trial_interval(
        &self,
        requested_end: f64,
        source_interval: Option<&CoupledLevelPoolSourceInterval>,
        source_plan: Option<&LevelPoolSourcePlan<'_, R>>,
    ) -> Result<CoupledLevelPoolTrial<R>, CoupledLevelPoolError> {
        let start = self.time();
        let config = &self.context.setup.config;
        if !requested_end.is_finite() || requested_end <= start {
            return Err(CoupledLevelPoolError::input(format!(
                "requested end={requested_end}, accepted time={start}: supply a finite later endpoint"
            )));
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
        let mut end = requested_end.min(start + config.maximum_step_s);
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
            match self.attempt(end, requested_end, retries, source_interval, source_plan) {
                Ok(next) => return Ok(CoupledLevelPoolTrial { next }),
                Err(error) => {
                    if !retryable(&error) {
                        return Err(error);
                    }
                    let shorter = start + dt * 0.5;
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
        source_interval: Option<&CoupledLevelPoolSourceInterval>,
        source_plan: Option<&LevelPoolSourcePlan<'_, R>>,
    ) -> Result<Self, CoupledLevelPoolError> {
        let prepared = source_plan.map(|plan| plan.prepare(end)).transpose()?;
        let sources = prepared.as_ref().map(|p| p.source_totals()).transpose()?;
        let super::iteration::SolvedInterval {
            input,
            evaluation,
            iterations,
            head_residual,
            transfer_residual,
        } = super::iteration::solve(&self.context, &self.pools, end, prepared.as_ref())?;
        let mass_residual = self.context.mass_check(
            &self.pools,
            &evaluation.pools,
            evaluation.exchange.aggregation_bound,
            sources.as_ref(),
        )?;
        let step = StoredStep {
            start_pools: self.pools.snapshot(),
            source_interval: source_interval.cloned(),
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
