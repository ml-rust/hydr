//! Common surface/channel trials: duration control, retries, and the owned candidate.

use super::accounting::within;
use super::batch::SurfaceChannelBatch;
use super::context::Context;
use super::evaluate::advanced;
use super::exchange::Exchange;
use super::forcing::BoundForcing;
use super::iteration::Stall;
use super::start::Start;
use super::state::{PreviousFrame, StoredStep};
use super::{SurfaceChannelError, SurfaceChannelForcingInterval, SurfaceChannelStepDiagnostics};
use crate::channel::{ChannelError, ChannelReach};
use hydr_solver::internal::CouplingPreparation;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};
use std::cell::Cell;
use std::sync::Arc;

/// Fraction of the smallest step limit a shortened attempt uses.
const LIMIT_SAFETY: f64 = 0.95;

/// Complete candidate of one common interval. Dropping it rejects every child together.
#[derive(Debug)]
pub struct SurfaceChannelTrial<R: Runtime<DType = DType>> {
    pub(super) next: SurfaceChannelBatch<R>,
}

impl<R: Runtime<DType = DType>> SurfaceChannelTrial<R> {
    /// Actual common endpoint after duration limits and recalculation, in seconds.
    pub fn time(&self) -> f64 {
        self.next.time()
    }

    /// Accepts the surface, every reach, and the link counters together.
    pub fn into_next(self) -> SurfaceChannelBatch<R> {
        self.next
    }

    /// Projects the candidate's common interval receipt.
    ///
    /// # Errors
    /// Returns runtime transfer errors and [`SurfaceChannelError::Numerical`] for a candidate
    /// without its receipt or with an unknown weir regime code.
    pub fn diagnostics(&self) -> Result<SurfaceChannelStepDiagnostics, SurfaceChannelError> {
        self.next.last_step()?.ok_or_else(|| {
            SurfaceChannelError::Numerical("candidate lacks its common interval receipt".into())
        })
    }
}

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    /// Calculates one common interval toward `requested_end_s` without changing this batch.
    ///
    /// The end is the smallest of `requested_end_s`, the start plus `maximum_step_s`, and the
    /// end of the covering surface forcing interval. An attempt above the surface or channel
    /// step limit restarts at 0.95 of the smallest limit. A stalled fixed point keeps `1/2`,
    /// `1/4`, or `1/8` of the interval by its measured contraction. Every other retryable
    /// rejection halves the common interval, or uses a shorter end a reach suggests.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Input`] for invalid forcing or an uncovered surface time.
    /// A caller policy at a ported end and a missing policy at an unported end are invalid
    /// forcing (`port-endpoint-policy`).
    /// Returns [`SurfaceChannelError::RecomputeInterval`] when caller integrated lateral volumes
    /// need a shorter span. Returns [`SurfaceChannelError::Unresolved`] when retries or the
    /// minimum step stop recalculation. Returns every non-retryable child rejection unchanged.
    pub fn trial_to(
        &self,
        requested_end_s: f64,
        forcing: &SurfaceChannelForcingInterval,
    ) -> Result<SurfaceChannelTrial<R>, SurfaceChannelError> {
        let setup = self.definition.setup();
        let config = &setup.config;
        let reach_ids: Vec<&str> = setup.reaches.iter().map(|r| r.id.as_str()).collect();
        let start = self.time_s;
        let requested = requested_end_s;
        let bound = BoundForcing::bind(&self.context, &reach_ids, (start, requested), forcing)?;
        let intervals = self.surface.coupling_forcing().intervals();
        let surface_end = intervals
            .get(intervals.partition_point(|interval| interval.end <= start))
            .map(|interval| interval.end)
            .ok_or_else(|| {
                SurfaceChannelError::input(format!(
                    "time={start}: the surface forcing schedule ends; supply surface forcing beyond the accepted time"
                ))
            })?;
        let minimum = config.minimum_step_s;
        let mut end = requested
            .min(start + config.maximum_step_s)
            .min(surface_end);
        if bound.requires_exact_span && end.to_bits() != requested.to_bits() {
            return Err(recompute(
                start,
                requested,
                end,
                minimum,
                0,
                "maximum_step_s or the surface forcing interval ends before the integrated span"
                    .into(),
            ));
        }
        let operands = self
            .context
            .start(&self.surface, &self.reaches, &reach_ids, end, bound)?;
        let mut retries = 0;
        let stalled = Cell::new(None);
        loop {
            let dt = end - start;
            if !dt.is_finite() || dt < minimum || end <= start {
                return Err(SurfaceChannelError::Unresolved {
                    dt_s: dt,
                    retries,
                    reason: "common duration falls below minimum_step_s".into(),
                });
            }
            stalled.set(None);
            let attempted = self.attempt(
                end,
                requested,
                retries,
                (&operands, &stalled),
                &reach_ids,
                forcing,
            );
            let error = match attempted {
                Ok(next) => return Ok(SurfaceChannelTrial { next }),
                Err(error) => error,
            };
            if !error.is_retryable() {
                return Err(error);
            }
            let suggested = match &error {
                SurfaceChannelError::RecomputeInterval {
                    suggested_end_s, ..
                }
                | SurfaceChannelError::Channel {
                    source:
                        ChannelError::RecomputeInterval {
                            suggested_end_s, ..
                        },
                    ..
                } => Some(*suggested_end_s),
                _ => None,
            };
            let fraction = stalled.get().map_or(0.5, |stall| {
                shortening(stall.contraction, config.contraction_target)
            });
            let shorter = suggested
                .filter(|&t| t.is_finite() && t > start && t < end)
                .unwrap_or(start + fraction * dt);
            if operands.forcing.requires_exact_span {
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
                return Err(SurfaceChannelError::Unresolved {
                    dt_s: dt,
                    retries,
                    reason: error.to_string(),
                });
            }
            end = shorter;
            retries += 1;
        }
    }

    /// Calculates one complete candidate ending at `end`.
    fn attempt(
        &self,
        end: f64,
        requested: f64,
        retries: usize,
        (operands, stalled): (&Start<R>, &Cell<Option<Stall>>),
        reach_ids: &[&str],
        forcing: &SurfaceChannelForcingInterval,
    ) -> Result<Self, SurfaceChannelError> {
        let context = &self.context;
        let config = &self.definition.setup().config;
        let start = self.time_s;
        let ghosts = context
            .ports
            .as_ref()
            .zip(operands.ports.as_ref())
            .map(|(topology, ports)| (&topology.surface, &ports.ghosts));
        let stage =
            match self
                .surface
                .coupling_prepare(end, retries, context.links.as_ref(), ghosts)?
            {
                CouplingPreparation::Ready(stage) => *stage,
                CouplingPreparation::SourceRetry(error)
                | CouplingPreparation::BoundaryRetry(error) => {
                    return Err(SurfaceChannelError::Numerical(format!(
                        "surface preparation: {error}. Shorten the common interval"
                    )));
                }
            };
        let ports = context.bind_ports(&stage, operands, &self.reaches, reach_ids)?;
        let limit = stage
            .limit()
            .min(operands.channel_limit_s)
            .min(ports.limit_s);
        if stage.dt() > limit {
            return Err(SurfaceChannelError::RecomputeInterval {
                start_s: start,
                requested_end_s: end,
                suggested_end_s: start + LIMIT_SAFETY * limit,
                reason: format!(
                    "surface CFL={}, surface depletion={}, channel start CFL or face depletion={}, ported reach CFL or face depletion={}",
                    stage.cfl_limit(),
                    stage.depletion_limit(),
                    operands.channel_limit_s,
                    ports.limit_s
                ),
            });
        }
        let dt = stage.dt();
        let mut slots: Vec<Option<ChannelReach<R>>> =
            (0..self.reaches.len()).map(|_| None).collect();
        let (surface, outcome, ports) = if context.link_count == 0 {
            let surface = context.complete_alone(&self.surface, &stage)?;
            (surface, LinkOutcome::empty(context)?, ports)
        } else {
            let attempt = context.attempt(operands, stage, ports)?;
            let solved = context.solve(
                config,
                &self.surface,
                &self.reaches,
                reach_ids,
                operands,
                &attempt,
                stalled,
            )?;
            let evaluation = solved.evaluation;
            for (entry, next) in context.linked.iter().zip(evaluation.linked) {
                slots[entry.reach] = Some(next);
            }
            let c = &context.client;
            let outcome = LinkOutcome {
                iterations: solved.iterations,
                contraction: solved.contraction,
                head_residual_m: solved.residuals.head_m,
                transfer_residual_m3: solved.residuals.transfer_m3,
                midpoint_surface_m: c.index_select(&evaluation.midpoint, 0, &context.link_a)?,
                midpoint_channel_m: c.index_select(&evaluation.midpoint, 0, &context.link_b)?,
                regime: evaluation.regime,
                exchange: evaluation.exchange,
            };
            (evaluation.surface, outcome, attempt.ports)
        };
        for &index in &context.unlinked {
            let reach = &self.reaches[index];
            let bound = &operands.forcing.reaches[index];
            let forcing = reach
                .bind_coupling_reserved(ports.reservation(operands, index)?, end, &bound.lateral)
                .map_err(|source| SurfaceChannelError::Channel {
                    reach_id: reach_ids[index].to_string(),
                    source,
                })?;
            slots[index] = Some(advanced(
                reach.trial_coupled(end, &forcing, None, None),
                start,
                end,
                reach_ids[index],
            )?);
        }
        let reaches = slots
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| SurfaceChannelError::Numerical("a reach lacks its candidate".into()))?;
        let steps = self.accepted_steps + 1;
        if surface.time().to_bits() != end.to_bits()
            || reaches
                .iter()
                .any(|r| r.time().to_bits() != end.to_bits() || r.accepted_steps() != steps)
        {
            return Err(SurfaceChannelError::Numerical(format!(
                "candidate children disagree on end={end} or accepted step {steps}"
            )));
        }
        let interval = context.port_interval(&ports, dt, &self.counters.ports)?;
        let exchange = outcome.exchange;
        let counters = context.advance_counters(&self.counters, &exchange.signed, &interval)?;
        let totals = context.step_totals(
            config,
            (&self.surface, &self.reaches),
            (&surface, &reaches),
            &exchange,
            (&self.counters.ports, &counters.ports, &interval),
        )?;
        let balance = context.balance(config, &surface, &reaches, &counters)?;
        if !within(balance.residual_m3, balance.bound_m3) {
            return Err(SurfaceChannelError::Numerical(format!(
                "lifetime composite water residual={}, bound={}: inspect child accounting",
                balance.residual_m3, balance.bound_m3
            )));
        }
        let step = StoredStep {
            previous: PreviousFrame {
                surface: self.surface.coupling_clone(),
                reaches: self.reaches.clone(),
                counters: self.counters.clone(),
            },
            forcing: forcing.clone(),
            start_s: start,
            end_s: end,
            requested_end_s: requested,
            retries,
            iterations: outcome.iterations,
            contraction: outcome.contraction,
            head_residual_m: outcome.head_residual_m,
            transfer_residual_m3: outcome.transfer_residual_m3,
            totals,
            midpoint_surface_m: outcome.midpoint_surface_m,
            midpoint_channel_m: outcome.midpoint_channel_m,
            signed_m3: exchange.signed,
            weir_m3: exchange.proposed,
            level: exchange.level,
            factor: exchange.factor,
            regime: outcome.regime,
            ports: interval,
        };
        Ok(Self {
            definition: Arc::clone(&self.definition),
            context: Arc::clone(&self.context),
            initial: Arc::clone(&self.initial),
            time_s: end,
            surface,
            reaches,
            accepted_steps: steps,
            counters,
            last: Some(Arc::new(step)),
        })
    }
}

/// Link results of one attempt: the converged fixed point, or nothing without links.
struct LinkOutcome<R: Runtime<DType = DType>> {
    iterations: usize,
    contraction: f64,
    head_residual_m: f64,
    transfer_residual_m3: f64,
    midpoint_surface_m: Tensor<R>,
    midpoint_channel_m: Tensor<R>,
    regime: Tensor<R>,
    exchange: Exchange<R>,
}

impl<R: Runtime<DType = DType>> LinkOutcome<R> {
    /// The outcome of a coordinator without links: no update and empty link tensors.
    fn empty(context: &Context<R>) -> Result<Self, SurfaceChannelError> {
        let empty = Tensor::<R>::zeros(&[0], DType::F64, context.client.device())?;
        Ok(Self {
            iterations: 0,
            contraction: 0.0,
            head_residual_m: 0.0,
            transfer_residual_m3: 0.0,
            midpoint_surface_m: empty.clone(),
            midpoint_channel_m: empty.clone(),
            regime: empty.clone(),
            exchange: Exchange::empty(&empty),
        })
    }
}

/// Smallest fraction of a stalled duration the next attempt keeps.
const STALL_FLOOR: f64 = 0.125;

/// Fraction of a stalled duration the next attempt keeps.
///
/// The largest of `1/2`, `1/4`, `1/8` whose product with `contraction` is at most `target`,
/// else `1/8`. Powers of two keep the rule exact.
fn shortening(contraction: f64, target: f64) -> f64 {
    let mut fraction = 0.5;
    while fraction > STALL_FLOOR && contraction * fraction > target {
        fraction *= 0.5;
    }
    fraction
}

/// Recompute request for a shorter integrated span, or `Unresolved` when none is admissible.
fn recompute(
    start: f64,
    requested: f64,
    suggested: f64,
    minimum: f64,
    retries: usize,
    reason: String,
) -> SurfaceChannelError {
    if !suggested.is_finite()
        || suggested <= start
        || suggested >= requested
        || suggested - start < minimum
    {
        SurfaceChannelError::Unresolved {
            dt_s: suggested - start,
            retries,
            reason,
        }
    } else {
        SurfaceChannelError::RecomputeInterval {
            start_s: start,
            requested_end_s: requested,
            suggested_end_s: suggested,
            reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortening_keeps_the_largest_power_of_two_under_the_target() {
        assert_eq!(shortening(0.9, 0.5), 0.5);
        assert_eq!(shortening(1.0, 0.5), 0.5);
        assert_eq!(shortening(1.2, 0.5), 0.25);
        assert_eq!(shortening(3.0, 0.5), 0.125);
        assert_eq!(shortening(100.0, 0.5), 0.125);
    }
}
