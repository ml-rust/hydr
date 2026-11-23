//! Common surface/channel trials: duration control, retries, and every attempt.

use super::accounting::{PoolInflowStep, within};
use super::batch::SurfaceChannelBatch;
use super::candidate::SurfaceChannelTrial;
use super::completion::advanced;
use super::evaluate::Accepted;
use super::forcing::BoundForcing;
use super::geometry::SurfaceChannelGeometryReceipt;
use super::iteration::Stall;
use super::operating::AttemptOperation;
use super::outcome::{Converged, LinkOutcome};
use super::pools::PoolChild;
use super::retry::{recompute, shortening};
use super::runoff::{
    RunoffStart, SurfaceRunoffStep, pool_receipts, runoff_delivery, runoff_end, runoff_start,
    with_runoff,
};
use super::start::Start;
use super::state::{PreviousFrame, StoredStep};
use super::{SurfaceChannelError, SurfaceChannelForcingInterval};
use crate::channel::{ChannelError, ChannelReach};
use hydr_solver::internal::CouplingPreparation;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};
use std::cell::Cell;
use std::sync::Arc;

/// Fraction of the smallest step limit a shortened attempt uses.
pub(super) const LIMIT_SAFETY: f64 = 0.95;

/// Accepted-start operands, stall cell, operation, and runoff start of one attempt.
type AttemptOperands<'a, R> = (
    &'a Start<R>,
    &'a Cell<Option<Stall>>,
    AttemptOperation<R>,
    Option<&'a RunoffStart>,
    Option<&'a PoolChild<R>>,
    &'a [SurfaceChannelGeometryReceipt],
);

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
    /// The end is the smallest of `requested_end_s`, the start plus `maximum_step_s`, the end of
    /// the covering surface forcing interval, and the next operating event. No interval then
    /// contains an event, and a retry only shortens it. An attempt above the surface or channel
    /// step limit restarts at 0.95 of the smallest limit. A stalled fixed point keeps `1/2`,
    /// `1/4`, or `1/8` of the interval by its measured contraction. Every other retryable
    /// rejection halves the common interval, or uses a shorter end a reach suggests.
    ///
    /// Pool sources hold one rate set per pool exactly when the pools declare source policies.
    /// Pool inflows hold one input per declared binding. An integrated pool inflow fixes the
    /// interval to the requested span, as a caller integrated lateral volume does.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Input`] for invalid forcing or an uncovered surface time.
    /// A caller policy at a ported end and a missing policy at an unported end are invalid
    /// forcing (`port-endpoint-policy`). A caller policy at a pool end is invalid forcing
    /// (`reach-end-ownership`). Pool forcing that differs from the declared pools is invalid.
    /// Returns [`SurfaceChannelError::Input`] without retry for:
    /// - a pool that fills past its top curve knot (`pool-capacity-exceeded`)
    /// - a running pump at or below suction at the start (`pump-without-suction`)
    /// - a running pump that delivers to a dry receiver without a declared outlet
    ///   (`pump-dry-delivery`)
    /// - a dry pool end at the accepted start (`pool-end-regime`)
    /// - a dry control signal node at a decision, and every operating child rejection
    ///
    /// Returns [`SurfaceChannelError::RecomputeInterval`] when caller integrated lateral or pool
    /// inflow volumes need a shorter span. Returns [`SurfaceChannelError::Unresolved`] when
    /// retries or the minimum step stop recalculation. Returns every non-retryable child
    /// rejection unchanged.
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
        let mut bound = BoundForcing::bind(&self.context, &reach_ids, (start, requested), forcing)?;
        let pool_start = self
            .context
            .pool_start(forcing, self.definition.pools.as_ref())?;
        if let Some(pool_start) = &pool_start {
            bound.requires_exact_span |= pool_start.inflows.has_interval_volumes();
        }
        let runoff = runoff_start(self, forcing, &reach_ids)?;
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
        let limited = requested
            .min(start + config.maximum_step_s)
            .min(surface_end);
        let mut end = self.operating_end(limited)?;
        if bound.requires_exact_span && end.to_bits() != requested.to_bits() {
            let reason = if end < limited {
                "an operating or geometry event lies inside the integrated span"
            } else {
                "maximum_step_s or the surface forcing interval ends before the integrated span"
            };
            return Err(recompute(start, requested, end, minimum, 0, reason.into()));
        }
        let (active_pools, pool_geometry) = match (&self.pools, self.definition.pools.as_ref()) {
            (Some(child), Some(declared)) if !setup.pool_curve_schedules.is_empty() => {
                let (next, receipts) = super::geometry::apply_pool_curves(
                    &self.context,
                    declared,
                    &setup.pool_curve_schedules,
                    child,
                    start,
                )?;
                (Some(next), receipts)
            }
            (Some(child), _) => (Some(child.clone()), Vec::new()),
            _ => (None, Vec::new()),
        };
        let pool_operands = match (&active_pools, pool_start) {
            (Some(child), Some(pool_start)) => Some((child, pool_start)),
            (None, None) => None,
            _ => {
                return Err(SurfaceChannelError::Numerical(
                    "the accepted pools differ from the pool forcing".into(),
                ));
            }
        };
        let children = (&self.surface, &self.reaches[..], pool_operands);
        let first = self.operation_at(end)?;
        let operands = self
            .context
            .start(children, &reach_ids, (end, &first.operations), bound)?;
        let mut prepared = Some(first);
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
            // A retry prepares the operating children again at its shorter end.
            let operation = match prepared.take() {
                Some(operation) if operation.end.to_bits() == end.to_bits() => operation,
                _ => self.operation_at(end)?,
            };
            let attempted = self.attempt(
                end,
                requested,
                retries,
                (
                    &operands,
                    &stalled,
                    operation,
                    runoff.as_ref(),
                    active_pools.as_ref(),
                    &pool_geometry,
                ),
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

    /// Calculates one complete candidate ending at `end` under `operation`.
    ///
    /// Runoff runs once, after the step limits pass and before the pools and the fixed point.
    fn attempt(
        &self,
        end: f64,
        requested: f64,
        retries: usize,
        (operands, stalled, mut operation, runoff_start, active_pools, pool_geometry): AttemptOperands<'_, R>,
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
                .coupling_prepare(end, retries, context.exchange_links(), ghosts)?
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
        let runoff = runoff_delivery(self, runoff_start, end)?;
        let mut pool_received = None;
        let dt = stage.dt();
        let steps = self.accepted_steps + 1;
        let mut slots: Vec<Option<ChannelReach<R>>> =
            (0..self.reaches.len()).map(|_| None).collect();
        // Candidate end stages and wet masks at every node, for decision samples.
        let mut nodes = None;
        let (surface, outcome, ports) = if context.transfer_count() == 0 {
            // Without transfers the surface still takes its runoff through the exchange.
            let lanes = runoff.as_ref().and_then(|runoff| runoff.surface.as_ref());
            let surface = context.complete_exchange(&self.surface, &stage, None, lanes)?;
            (surface, LinkOutcome::empty(context)?, ports)
        } else {
            let pool_child = active_pools.map(|child| (child, steps));
            let links = Arc::clone(&operation.operations);
            let attempt =
                context.attempt(operands, (stage, links), ports, pool_child, runoff.as_ref())?;
            pool_received = pool_receipts(self, &attempt)?;
            let accepted = Accepted {
                surface: &self.surface,
                reaches: &self.reaches,
                pools: active_pools,
            };
            let solved = context.solve(config, accepted, reach_ids, operands, &attempt, stalled)?;
            let converged = Converged::new(context, solved, attempt.pools.as_ref())?;
            nodes = Some(converged.nodes);
            for (entry, next) in context.linked.iter().zip(converged.linked) {
                slots[entry.reach] = Some(next);
            }
            let ended = context.pool_ends.as_ref().map_or(&[][..], |e| &e.ended);
            for (&reach, next) in ended.iter().zip(converged.ended) {
                slots[reach] = Some(next);
            }
            (converged.surface, converged.outcome, attempt.ports)
        };
        for &index in &context.unlinked {
            let reach = &self.reaches[index];
            let bound = &operands.forcing.reaches[index];
            let inputs = with_runoff(&bound.lateral, runoff.as_ref(), index);
            let forcing = reach
                .bind_coupling_reserved(ports.reservation(operands, index)?, end, &inputs)
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
        let (pools, pool_inflow) = match outcome.pools {
            Some((child, inflow)) => (Some(child), inflow),
            None => (None, PoolInflowStep::default()),
        };
        if pools.is_some() != self.pools.is_some() {
            return Err(SurfaceChannelError::Numerical(
                "the candidate lacks its pools".into(),
            ));
        }
        if surface.time().to_bits() != end.to_bits()
            || reaches
                .iter()
                .any(|r| r.time().to_bits() != end.to_bits() || r.accepted_steps() != steps)
            || pools.as_ref().is_some_and(|p| {
                p.batch.time().to_bits() != end.to_bits() || p.batch.accepted_steps() != steps
            })
        {
            return Err(SurfaceChannelError::Numerical(format!(
                "candidate children disagree on end={end} or accepted step {steps}"
            )));
        }
        let runoff_step = runoff_end(
            self,
            (runoff_start, runoff.as_ref()),
            (&surface, &reaches, pools.as_ref()),
            pool_received.as_ref(),
            reach_ids,
            end,
        )?;
        let interval = context.port_interval(&ports, dt, &self.counters.ports)?;
        let pool_ends = outcome.pool_ends;
        context.check_end_pairing(&reaches, reach_ids, &pool_ends.volume_m3)?;
        let exchange = outcome.exchange;
        let counters = context.advance_counters(
            &self.counters,
            &exchange.signed,
            (&interval, &pool_ends),
            runoff.as_ref(),
        )?;
        let surface_runoff = runoff
            .as_ref()
            .map_or_else(SurfaceRunoffStep::default, |r| r.surface_step);
        let next_runoff = runoff.map_or_else(|| self.runoff.clone(), |r| Some(r.next));
        let totals = context.step_totals(
            config,
            (
                (&self.surface, &self.reaches, self.pools.as_ref()),
                (&surface, &reaches, pools.as_ref()),
            ),
            &exchange,
            (&self.counters, &counters, &interval),
            (pool_inflow, &pool_ends),
            ([self.runoff.as_ref(), next_runoff.as_ref()], surface_runoff),
        )?;
        let balance = context.balance(
            config,
            (&surface, &reaches, pools.as_ref()),
            &counters,
            next_runoff.as_ref(),
        )?;
        if !within(balance.residual_m3, balance.bound_m3) {
            return Err(SurfaceChannelError::Numerical(format!(
                "lifetime composite water residual={}, bound={}: inspect child accounting",
                balance.residual_m3, balance.bound_m3
            )));
        }
        let mut geometry = std::mem::take(&mut operation.geometry);
        geometry.extend_from_slice(pool_geometry);
        for receipt in &mut geometry {
            receipt.storage_before_m3 = totals.storage_before_m3;
            receipt.storage_after_m3 = totals.storage_before_m3;
        }
        let (operating, operation) = self.complete_operating(operation, end, nodes.as_ref())?;
        let step = StoredStep {
            previous: PreviousFrame {
                surface: self.surface.coupling_clone(),
                reaches: self.reaches.clone(),
                counters: self.counters.clone(),
                pools: self.pools.clone(),
                operating: self.operating.clone(),
                runoff: self.runoff.clone(),
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
            pool_ends,
            operation,
            geometry,
            runoff: runoff_step,
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
            pools,
            operating,
            runoff: next_runoff,
        })
    }
}
