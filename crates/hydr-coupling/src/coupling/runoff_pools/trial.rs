use super::{
    RunoffPoolBatch, RunoffPoolError, RunoffPoolForcingInterval, RunoffPoolStepDiagnostics,
};
use crate::coupling::CoupledLevelPoolBatch;
use crate::coupling::CoupledLevelPoolError;
use crate::runoff::{SubcatchmentBatch, SubcatchmentError, SubcatchmentInput};
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

/// Complete common candidate. Dropping it preserves both accepted child engines.
#[derive(Debug)]
pub struct RunoffPoolTrial<R: Runtime<DType = DType>> {
    next: RunoffPoolBatch<R>,
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
    /// Recalculate both children from accepted starts until one common interval passes every gate.
    pub fn trial_with_forcing_to(
        &self,
        requested_end: f64,
        forcing: &RunoffPoolForcingInterval,
    ) -> Result<RunoffPoolTrial<R>, RunoffPoolError> {
        self.context
            .check_forcing(forcing, self.time(), requested_end)?;
        let inputs = self.context.runoff_inputs(forcing);
        let start = self.time();
        let mut end = requested_end.min(start + self.context.maximum);
        let mut retries = 0;
        loop {
            let dt = end - start;
            if !dt.is_finite() || dt < self.context.minimum || end <= start {
                return Err(RunoffPoolError::Unresolved {
                    dt_s: dt,
                    retries,
                    reason: "common interval cannot represent the effective minimum duration"
                        .into(),
                });
            }
            match self.attempt(end, requested_end, retries, forcing, &inputs) {
                Ok(next) => return Ok(RunoffPoolTrial { next }),
                Err(error) => {
                    let suggested = match &error {
                        RunoffPoolError::Pools(CoupledLevelPoolError::RecomputeInterval {
                            start_s,
                            requested_end_s,
                            suggested_end_s,
                            ..
                        }) => {
                            if start_s.to_bits() != start.to_bits()
                                || requested_end_s.to_bits() != end.to_bits()
                            {
                                return Err(RunoffPoolError::input(
                                    "pool recalculation request differs from the original common attempt",
                                ));
                            }
                            Some(*suggested_end_s)
                        }
                        RunoffPoolError::Runoff(SubcatchmentError::Numerical { .. })
                        | RunoffPoolError::Numerical(_) => Some(start + dt * 0.5),
                        RunoffPoolError::Pools(CoupledLevelPoolError::Unresolved { .. }) => None,
                        _ => return Err(error),
                    };
                    let suggested = match suggested {
                        Some(value) => value,
                        None => {
                            return Err(RunoffPoolError::Unresolved {
                                dt_s: dt,
                                retries,
                                reason: error.to_string(),
                            });
                        }
                    };
                    if retries >= self.setup().config.retry_limit
                        || !suggested.is_finite()
                        || suggested <= start
                        || suggested >= end
                        || suggested - start < self.context.minimum
                    {
                        return Err(RunoffPoolError::Unresolved {
                            dt_s: dt,
                            retries,
                            reason: error.to_string(),
                        });
                    }
                    retries += 1;
                    end = suggested;
                }
            }
        }
    }
    fn attempt(
        &self,
        end: f64,
        requested: f64,
        retries: usize,
        forcing: &RunoffPoolForcingInterval,
        inputs: &[SubcatchmentInput],
    ) -> Result<Self, RunoffPoolError> {
        let runoff_trial = self.runoff.trial_to(end, inputs)?;
        let volumes = self.context.route(runoff_trial.original_outlet()?)?;
        let pool_trial = self.pools.trial_with_native_inflows_to(
            end,
            forcing.pool_sources.as_deref(),
            &volumes,
        )?;
        let mut next = Self {
            context: Arc::clone(&self.context),
            runoff: runoff_trial.into_next(),
            pools: pool_trial.into_next(),
            last: None,
        };
        let original = next.pools.original_inflow_volumes()?.ok_or_else(|| {
            RunoffPoolError::numerical("receiving candidate lacks original integrated inflows")
        })?;
        next.context
            .check_pair(next.runoff.original_outlet()?, &original, false)?;
        next.ledger()?;
        let balance = next.interval_balance()?;
        next.last = Some(Arc::new(RunoffPoolStepDiagnostics {
            start_s: self.time(),
            end_s: end,
            requested_end_s: requested,
            retries,
            forcing: forcing.clone(),
            outlets: self.setup().outlets.clone(),
            balance,
        }));
        Ok(next)
    }
}
impl<R: Runtime<DType = DType>> RunoffPoolTrial<R>
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
    /// Accept runoff, native soil, routing, and pools together by ownership.
    pub fn into_next(self) -> RunoffPoolBatch<R> {
        self.next
    }
    /// Actual common candidate endpoint, in seconds.
    pub fn time(&self) -> f64 {
        self.next.time()
    }
    /// Read-only complete candidate runoff engine.
    pub fn runoff(&self) -> &SubcatchmentBatch<R> {
        self.next.runoff()
    }
    /// Read-only complete candidate receiving pool engine.
    pub fn pools(&self) -> &CoupledLevelPoolBatch<R> {
        self.next.pools()
    }
    /// Complete common candidate receipt with derived physical accounting.
    pub fn diagnostics(&self) -> Result<RunoffPoolStepDiagnostics, RunoffPoolError> {
        self.next.last_step()?.ok_or_else(|| {
            RunoffPoolError::numerical("common candidate lacks its accepted interval receipt")
        })
    }
}
