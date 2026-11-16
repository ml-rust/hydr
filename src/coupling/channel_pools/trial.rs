use super::batch::ChannelPoolBatch;
use super::state::{Accepted, BoundForcing, PreviousFrame, StoredStep};
use super::{
    ChannelPoolError, ChannelPoolForcingInterval, ChannelPoolStepDiagnostics,
    ChannelPoolTensorForcingInterval,
};
use crate::channel::ChannelError;
use crate::storage::LevelPoolError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{AndersonAlgorithms, BatchedBisectionAlgorithms};
use std::sync::Arc;

/// Complete immutable candidate with a shared channel/pool acceptance boundary.
#[derive(Debug)]
pub struct ChannelPoolTrial<R: Runtime<DType = DType>> {
    pub(super) next: ChannelPoolBatch<R>,
}
impl<R: Runtime<DType = DType>> ChannelPoolTrial<R> {
    /// Actual common endpoint after complete interval recalculations.
    pub fn time(&self) -> f64 {
        self.next.time()
    }
    /// Accept every child and water counter together by ownership.
    pub fn into_next(self) -> ChannelPoolBatch<R> {
        self.next
    }
}
impl<R: Runtime<DType = DType>> ChannelPoolTrial<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
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
        + BatchedBisectionAlgorithms<R>
        + AndersonAlgorithms<R>,
{
    /// Project the candidate's exact final common interval receipt.
    pub fn diagnostics(&self) -> Result<ChannelPoolStepDiagnostics, ChannelPoolError> {
        self.next.last_step()?.ok_or_else(|| {
            ChannelPoolError::numerical("candidate lacks its final common interval receipt")
        })
    }
}
impl<R: Runtime<DType = DType>> ChannelPoolBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
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
        + BatchedBisectionAlgorithms<R>
        + AndersonAlgorithms<R>
        + numr::ops::TensorOps<R>
        + numr::ops::NextRepresentableOps<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    /// Calculate a common candidate from explicit external endpoint policies and lateral inputs.
    pub fn trial_to(
        &self,
        requested_end_s: f64,
        forcing: &ChannelPoolForcingInterval,
    ) -> Result<ChannelPoolTrial<R>, ChannelPoolError> {
        self.trial_bound(
            requested_end_s,
            self.bind_host_forcing(requested_end_s, forcing)?,
        )
    }
    /// Copy caller native operands once, then calculate a complete common candidate.
    pub fn trial_tensor_to(
        &self,
        requested_end_s: f64,
        forcing: &ChannelPoolTensorForcingInterval<R>,
    ) -> Result<ChannelPoolTrial<R>, ChannelPoolError> {
        self.trial_bound(
            requested_end_s,
            self.bind_tensor_forcing(requested_end_s, forcing)?,
        )
    }
    pub(super) fn trial_bound(
        &self,
        requested: f64,
        forcing: Arc<BoundForcing<R>>,
    ) -> Result<ChannelPoolTrial<R>, ChannelPoolError> {
        super::forcing::check_span(self.time(), requested, forcing.start_s, forcing.end_s)?;
        self.context
            .check_epochs(&self.accepted.reaches, &self.accepted.pools)?;
        let config = &self.setup().config;
        let start = self.time();
        let (minimum, maximum) = self.context.duration_limits();
        let mut end = requested.min(start + maximum);
        if forcing.requires_exact_span && end.to_bits() != requested.to_bits() {
            return Err(recompute(
                start,
                requested,
                end,
                minimum,
                0,
                "requested duration exceeds maximum_step_s".into(),
            ));
        }
        let mut retries = 0;
        loop {
            let dt = end - start;
            if !dt.is_finite() || dt < minimum || end <= start {
                return Err(ChannelPoolError::Unresolved {
                    dt_s: dt,
                    retries,
                    reason: "common duration cannot represent an admitted minimum step".into(),
                });
            }
            match self.attempt(end, requested, retries, Arc::clone(&forcing)) {
                Ok(next) => return Ok(ChannelPoolTrial { next }),
                Err(error) => {
                    if !retryable(&error) {
                        return Err(error);
                    }
                    let suggested = match &error {
                        ChannelPoolError::RecomputeInterval {
                            suggested_end_s, ..
                        } => Some(*suggested_end_s),
                        ChannelPoolError::Channel {
                            source:
                                ChannelError::RecomputeInterval {
                                    suggested_end_s, ..
                                },
                            ..
                        } => Some(*suggested_end_s),
                        _ => None,
                    };
                    let shorter = suggested
                        .filter(|&t| t.is_finite() && t > start && t < end)
                        .unwrap_or(start + dt * 0.5);
                    if forcing.requires_exact_span {
                        return Err(recompute(
                            start,
                            requested,
                            shorter,
                            minimum,
                            retries,
                            error.to_string(),
                        ));
                    }
                    if retries >= config.retry_limit
                        || shorter - start < minimum
                        || shorter <= start
                        || shorter >= end
                    {
                        return Err(ChannelPoolError::Unresolved {
                            dt_s: dt,
                            retries,
                            reason: error.to_string(),
                        });
                    }
                    end = shorter;
                    retries += 1;
                }
            }
        }
    }
    fn attempt(
        &self,
        end: f64,
        requested: f64,
        retries: usize,
        forcing: Arc<BoundForcing<R>>,
    ) -> Result<Self, ChannelPoolError> {
        let solved = super::iteration::solve(&self.context, &self.accepted, end, &forcing)?;
        let evaluation = solved.evaluation;
        let accounting = self
            .context
            .interval_accounting(&self.accepted, &evaluation)?;
        let counters = self.context.advance_counters(
            &self.accepted.counters,
            &evaluation.interval,
            &accounting,
        )?;
        let previous = PreviousFrame {
            reaches: self.accepted.reaches.iter().map(|r| r.snapshot()).collect(),
            pools: self.accepted.pools.snapshot(),
            counters: self.accepted.counters.clone(),
        };
        let next = Self {
            context: Arc::clone(&self.context),
            accepted: Accepted {
                reaches: evaluation.reaches,
                pools: evaluation.pools,
                counters,
                last: Some(Arc::new(StoredStep {
                    previous,
                    forcing,
                    requested_end_s: requested,
                    iterations: solved.iterations,
                    retries,
                    interval: evaluation.interval,
                    accounting,
                })),
            },
        };
        next.context
            .check_epochs(&next.accepted.reaches, &next.accepted.pools)?;
        next.context.water_balance(&next.accepted)?;
        Ok(next)
    }
}
fn retryable(error: &ChannelPoolError) -> bool {
    matches!(
        error,
        ChannelPoolError::Numerical(_)
            | ChannelPoolError::RecomputeInterval { .. }
            | ChannelPoolError::Pool(LevelPoolError::Numerical { .. })
            | ChannelPoolError::Channel {
                source: ChannelError::Numerical { .. }
                    | ChannelError::Unresolved { .. }
                    | ChannelError::RecomputeInterval { .. },
                ..
            }
    )
}
fn recompute(
    start: f64,
    requested: f64,
    suggested: f64,
    minimum: f64,
    retries: usize,
    reason: String,
) -> ChannelPoolError {
    if !suggested.is_finite()
        || suggested <= start
        || suggested >= requested
        || suggested - start < minimum
    {
        ChannelPoolError::Unresolved {
            dt_s: suggested - start,
            retries,
            reason,
        }
    } else {
        ChannelPoolError::RecomputeInterval {
            start_s: start,
            requested_end_s: requested,
            suggested_end_s: suggested,
            reason,
        }
    }
}
