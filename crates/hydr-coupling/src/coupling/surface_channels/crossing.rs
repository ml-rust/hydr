//! Cutoff and wet-support crossing localization via `solvr` certified bisection.
//!
//! When an attempted interval crosses a raised attachment bed, law cutoff, pool cut, pool-end
//! wet support, or orifice crown, `localize_crossing` brackets and locates the crossing time
//! from the accepted start using complete trial probes. Integrated inputs return
//! `RecomputeInterval` at the linear crossing estimate instead of probing.

use super::SurfaceChannelError;
use super::batch::SurfaceChannelBatch;
use super::candidate::SurfaceChannelTrial;
use super::context::{Context, Ops};
use super::pools::PoolCurveState;
use super::retry::{CROSSING_TIME_RTOL, linear_crossing_estimate, recompute};
use super::trial::AttemptOperands;
use crate::coupling::surface_channels::SurfaceChannelForcingInterval;
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
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootProbe,
    CertifiedRootQuery, CertifiedRootValues,
};
use std::cell::{Cell, RefCell};

/// Maximum certified bisection iterations for one crossing localization.
const MAX_CROSSING_ITERS: usize = 32;

/// Signed margin recorded when a converged candidate fails a cutoff or wet-support check.
#[derive(Clone, Copy, Debug)]
pub(super) struct CrossingSignal {
    /// Nonpositive margin (`stage - cutoff` or `available - demand`), or `-1.0` when unavailable.
    pub margin: f64,
}

/// Returns true when `error` is a retryable cutoff, pool-cut, wet-support, or orifice-crown
/// crossing rejection that localization handles instead of blind halving.
pub(super) fn is_crossing_error(error: &SurfaceChannelError) -> bool {
    let SurfaceChannelError::Numerical(msg) = error else {
        return false;
    };
    msg.contains("active transfer crosses a raised local bed or law cutoff")
        || msg.contains("connection exports exceed original water above an attachment cut")
        || msg.contains("active connection loses wet support during the common interval")
        || (msg.contains("link weir:") && msg.contains("crown"))
}

/// Linear crossing estimate for integrated-forcing `RecomputeInterval` suggestions.
pub(super) fn suggest_crossing_end(
    start: f64,
    end: f64,
    start_margin: f64,
    signal: Option<CrossingSignal>,
) -> f64 {
    let end_margin = signal.map_or(-1.0, |s| s.margin);
    linear_crossing_estimate(start, end, start_margin, end_margin)
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Minimum signed `end_stage - cutoff` across all active links whose donor has a cutoff
    /// raised above its pool origin, or `None` when no such link is active.
    pub(super) fn raised_cutoff_margin(
        &self,
        curves: &PoolCurveState<R>,
        end_stage: &Tensor<R>,
        signed: &Tensor<R>,
    ) -> Result<Option<f64>, SurfaceChannelError> {
        let Some(topology) = &self.pools else {
            return Ok(None);
        };
        if self.link_count == 0 {
            return Ok(None);
        }
        let c = &self.client;
        let a_stage = c
            .index_select(end_stage, 0, &self.link_a)?
            .try_to_vec::<f64>()?;
        let b_stage = c
            .index_select(end_stage, 0, &self.link_b)?
            .try_to_vec::<f64>()?;
        let cut_a = topology.cutoff_a.try_to_vec::<f64>()?;
        let cut_b = topology.cutoff_b.try_to_vec::<f64>()?;
        let orig_a = c
            .index_select(&curves.group_origin, 0, &self.link_a)?
            .try_to_vec::<f64>()?;
        let orig_b = c
            .index_select(&curves.group_origin, 0, &self.link_b)?
            .try_to_vec::<f64>()?;
        let vol = signed.try_to_vec::<f64>()?;
        let mut min_margin: Option<f64> = None;
        for i in 0..self.link_count {
            if vol[i] > 0.0 && cut_a[i] > orig_a[i] {
                let m = a_stage[i] - cut_a[i];
                min_margin = Some(min_margin.map_or(m, |cur| cur.min(m)));
            }
            if vol[i] < 0.0 && cut_b[i] > orig_b[i] {
                let m = b_stage[i] - cut_b[i];
                min_margin = Some(min_margin.map_or(m, |cur| cur.min(m)));
            }
        }
        Ok(min_margin)
    }

    /// Minimum positive `start_stage - cutoff` across all raised-cutoff pool attachments currently
    /// strictly above their cutoff, defaulting to `1.0`.
    pub(super) fn start_cutoff_margin(
        &self,
        curves: Option<&PoolCurveState<R>>,
        start_stage: &Tensor<R>,
    ) -> Result<f64, SurfaceChannelError> {
        let (Some(topology), Some(curves)) = (&self.pools, curves) else {
            return Ok(1.0);
        };
        if self.link_count == 0 {
            return Ok(1.0);
        }
        let c = &self.client;
        let a_stage = c
            .index_select(start_stage, 0, &self.link_a)?
            .try_to_vec::<f64>()?;
        let b_stage = c
            .index_select(start_stage, 0, &self.link_b)?
            .try_to_vec::<f64>()?;
        let cut_a = topology.cutoff_a.try_to_vec::<f64>()?;
        let cut_b = topology.cutoff_b.try_to_vec::<f64>()?;
        let orig_a = c
            .index_select(&curves.group_origin, 0, &self.link_a)?
            .try_to_vec::<f64>()?;
        let orig_b = c
            .index_select(&curves.group_origin, 0, &self.link_b)?
            .try_to_vec::<f64>()?;
        let mut min_pos: Option<f64> = None;
        for i in 0..self.link_count {
            if cut_a[i] > orig_a[i] && a_stage[i] > cut_a[i] {
                let m = a_stage[i] - cut_a[i];
                min_pos = Some(min_pos.map_or(m, |cur| cur.min(m)));
            }
            if cut_b[i] > orig_b[i] && b_stage[i] > cut_b[i] {
                let m = b_stage[i] - cut_b[i];
                min_pos = Some(min_pos.map_or(m, |cur| cur.min(m)));
            }
        }
        Ok(min_pos.unwrap_or(1.0).max(f64::MIN_POSITIVE))
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
        + CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    /// Resolves a crossing rejection: returns `RecomputeInterval` for integrated inputs, else
    /// localizes the crossing time via certified bisection.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resolve_crossing(
        &self,
        failed_end: f64,
        requested: f64,
        retries: usize,
        signal: Option<CrossingSignal>,
        error: &SurfaceChannelError,
        operands: &super::start::Start<R>,
        runoff_start: Option<&super::runoff::RunoffStart>,
        start_geometry: &super::geometry::StartGeometry<R>,
        reach_ids: &[&str],
        forcing: &SurfaceChannelForcingInterval,
    ) -> Result<SurfaceChannelTrial<R>, SurfaceChannelError> {
        let active_surface = start_geometry.surface.as_ref().unwrap_or(&self.surface);
        let active_reaches = start_geometry.reaches.as_deref().unwrap_or(&self.reaches);
        let active_pools = start_geometry.pools.as_ref();
        let extra_geometry = &start_geometry.receipts;
        let config = &self.definition.setup().config;
        let start = self.time_s;
        let start_m = self
            .context
            .start_cutoff_margin(active_pools.map(|p| p.curves.as_ref()), &operands.stage)?;
        if operands.forcing.requires_exact_span {
            let est = suggest_crossing_end(start, failed_end, start_m, signal);
            return Err(recompute(
                start,
                requested,
                est,
                config.minimum_step_s,
                retries,
                error.to_string(),
            ));
        }
        let end_m = signal.map_or(-1.0, |s| s.margin);
        if let Some(trial) = self.localize_crossing(
            failed_end,
            requested,
            retries,
            start_m,
            end_m,
            operands,
            runoff_start,
            active_surface,
            active_reaches,
            active_pools,
            extra_geometry,
            reach_ids,
            forcing,
        )? {
            return Ok(trial);
        }
        Err(SurfaceChannelError::Unresolved {
            dt_s: failed_end - start,
            retries: config.retry_limit,
            reason: error.to_string(),
        })
    }

    /// Locates the crossing time in `(start, failed_end)` using `solvr` certified bisection.
    ///
    /// Returns `Ok(Some(trial))` when a valid candidate at the lower end of the certified bracket
    /// is found above `minimum_step_s`, or `Ok(None)` when localization cannot find an admissible
    /// candidate above `minimum_step_s`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn localize_crossing(
        &self,
        failed_end: f64,
        requested: f64,
        base_retries: usize,
        start_margin: f64,
        failed_margin: f64,
        operands: &super::start::Start<R>,
        runoff_start: Option<&super::runoff::RunoffStart>,
        active_surface: &crate::solver::Solver<R>,
        active_reaches: &[crate::channel::ChannelReach<R>],
        active_pools: Option<&super::pools::PoolChild<R>>,
        extra_geometry: &[super::geometry::SurfaceChannelGeometryReceipt],
        reach_ids: &[&str],
        forcing: &SurfaceChannelForcingInterval,
    ) -> Result<Option<SurfaceChannelTrial<R>>, SurfaceChannelError> {
        let config = &self.definition.setup().config;
        let start = self.time_s;
        let budget = config.retry_limit.saturating_sub(base_retries);
        if budget == 0 {
            return Ok(None);
        }
        let max_iters = budget.min(MAX_CROSSING_ITERS);
        let c = &self.context.client;
        let device = c.device();
        let lo = Tensor::<R>::from_slice(&[start], &[1], device)?;
        let hi = Tensor::<R>::from_slice(&[failed_end], &[1], device)?;
        let slope = Tensor::<R>::from_slice(&[1e-300], &[1], device)?;
        let options = CertifiedBisectionOptions {
            atol: config.minimum_step_s,
            rtol: CROSSING_TIME_RTOL,
            max_iterations: max_iters,
            max_points: 2 * (max_iters + 1),
            probe: CertifiedRootProbe::SlackSecant,
        };

        let probes = Cell::new(0usize);
        let best: RefCell<Option<(f64, Self)>> = RefCell::new(None);
        let fatal: RefCell<Option<SurfaceChannelError>> = RefCell::new(None);
        let stalled = Cell::new(None);
        let crossing = Cell::new(None);

        // Residual g(t) = cutoff - stage(t): strictly negative at `start`, positive at `failed_end`.
        let evaluate_query =
            |query: &CertifiedRootQuery<R>| -> numr::error::Result<CertifiedRootValues<R>> {
                let points = query.point.try_to_vec::<f64>()?;
                let mut vals = Vec::with_capacity(points.len());
                for &t in &points {
                    if t.to_bits() == start.to_bits() {
                        vals.push(-start_margin.max(f64::MIN_POSITIVE));
                        continue;
                    }
                    if t.to_bits() == failed_end.to_bits() {
                        vals.push((-failed_margin).max(f64::MIN_POSITIVE));
                        continue;
                    }
                    let next_probe = probes.get() + 1;
                    probes.set(next_probe);
                    stalled.set(None);
                    crossing.set(None);
                    let operation = match self.operation_at(t) {
                        Ok(op) => op,
                        Err(e) => {
                            *fatal.borrow_mut() = Some(e);
                            return Err(numr::error::Error::InvalidArgument {
                                arg: "crossing-localization",
                                reason: "operation preparation failed".into(),
                            });
                        }
                    };
                    let args: AttemptOperands<'_, R> = (
                        operands,
                        &stalled,
                        &crossing,
                        operation,
                        runoff_start,
                        active_surface,
                        active_reaches,
                        active_pools,
                        extra_geometry,
                    );
                    match self.attempt(
                        t,
                        requested,
                        base_retries + next_probe,
                        args,
                        reach_ids,
                        forcing,
                    ) {
                        Ok(candidate) => {
                            let pos = match candidate_positive_margin(&candidate) {
                                Ok(m) => m.max(f64::MIN_POSITIVE),
                                Err(e) => {
                                    *fatal.borrow_mut() = Some(e);
                                    return Err(numr::error::Error::InvalidArgument {
                                        arg: "crossing-localization",
                                        reason: "margin read failed".into(),
                                    });
                                }
                            };
                            let mut slot = best.borrow_mut();
                            if slot.as_ref().is_none_or(|(bt, _)| t > *bt) {
                                *slot = Some((t, candidate));
                            }
                            vals.push(-pos);
                        }
                        Err(err) if err.is_retryable() => {
                            let neg = crossing
                                .get()
                                .map_or(-1.0, |s| s.margin.min(-f64::MIN_POSITIVE));
                            vals.push(-neg);
                        }
                        Err(err) => {
                            *fatal.borrow_mut() = Some(err);
                            return Err(numr::error::Error::InvalidArgument {
                                arg: "crossing-localization",
                                reason: "non-retryable error during crossing localization".into(),
                            });
                        }
                    }
                }
                let tensor = Tensor::<R>::from_slice(&vals, &[vals.len()], device)?;
                Ok(CertifiedRootValues {
                    lower: tensor.clone(),
                    upper: tensor,
                })
            };

        let result = c.bisect_certified(evaluate_query, &lo, &hi, &slope, &options);
        if let Some(err) = fatal.into_inner() {
            return Err(err);
        }
        result
            .map_err(|e| SurfaceChannelError::Numerical(format!("crossing localization: {e}")))?;
        let found = best.into_inner();
        match found {
            Some((t, candidate)) if t - start >= config.minimum_step_s => {
                Ok(Some(SurfaceChannelTrial { next: candidate }))
            }
            _ => Ok(None),
        }
    }
}

/// Positive signed margin (`stage - cutoff`) of an accepted candidate, defaulting to `1.0` when
/// no raised-cutoff pool attachment is active.
fn candidate_positive_margin<R: Runtime<DType = DType>>(
    candidate: &SurfaceChannelBatch<R>,
) -> Result<f64, SurfaceChannelError>
where
    R::Client: Ops<R>,
{
    let (Some(pools), Some(step)) = (&candidate.pools, &candidate.last) else {
        return Ok(1.0);
    };
    let (pool_stage, _) = candidate.context.pool_stages(&pools.batch)?;
    let reach_ids: Vec<&str> = candidate
        .definition
        .setup()
        .reaches
        .iter()
        .map(|r| r.id.as_str())
        .collect();
    let borrowed: Vec<&crate::channel::ChannelReach<R>> = candidate
        .context
        .linked
        .iter()
        .map(|e| &candidate.reaches[e.reach])
        .collect();
    let (sc_stage, _) = candidate
        .context
        .stages(&candidate.surface, &borrowed, &reach_ids)?;
    let end_stage = candidate.context.client.cat(&[&sc_stage, &pool_stage], 0)?;
    Ok(candidate
        .context
        .raised_cutoff_margin(&pools.curves, &end_stage, &step.signed_m3)?
        .unwrap_or(1.0)
        .max(f64::MIN_POSITIVE))
}
