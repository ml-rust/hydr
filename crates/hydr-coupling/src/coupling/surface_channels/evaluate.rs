//! One fixed-point map: midpoint heads, link volumes, levelling caps, donor limits, and every
//! child completion.
//!
//! Every evaluation starts from the accepted children, so no trial state carries over.
//! A child trial whose exchange operands repeat an earlier trial of the attempt in every bit
//! takes that trial's result. See [`TrialMemo`].
//! Momentum closure: surface-to-channel water enters the reach with the surface donor's
//! post-transport velocity projected on the reach tangent. Channel-to-surface water leaves at
//! the channel's accepted-start velocity, the channel's own withdrawal rule, and enters the
//! surface along the tangent. Several inflows into one surface cell carry their summed momentum.
//! Structure links hold a zero tangent, so their water enters either side without velocity.
//! Pools hold no momentum.

use super::SurfaceChannelError;
use super::context::{Context, Ops, transfer_error};
use super::exchange::Exchange;
use super::levelling::Levelling;
use super::memo::{TrialKeys, TrialMemo};
use super::pools::{PoolAttempt, PoolChild, PoolCompletion, charge};
use super::ports::PortBinding;
use super::start::Start;
use crate::channel::ChannelReach;
use crate::numerics::water_transfers::AggregationKind;
use crate::solver::Solver;
use crate::storage::LevelPoolBatch;
use hydr_channel::internal::ChannelCouplingTrial;
use hydr_solver::internal::{CouplingCompletion, ExchangeSource, SurfaceCouplingStage};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ScalarOps, ShapeOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Relative donor margin per withdrawal binding, in units of `f64::EPSILON`.
const SLOT_MARGIN_ULPS: f64 = 4.0;

/// Slack bindings added to every slot margin.
const SLOT_MARGIN_BASE: f64 = 8.0;

/// Accepted children every evaluation of one trial starts from.
pub(super) struct Accepted<'a, R: Runtime<DType = DType>> {
    pub surface: &'a Solver<R>,
    /// Reaches in setup order.
    pub reaches: &'a [ChannelReach<R>],
    pub pools: Option<&'a PoolChild<R>>,
}

impl<R: Runtime<DType = DType>> Clone for Accepted<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime<DType = DType>> Copy for Accepted<'_, R> {}

/// Operands of one attempted common duration.
pub(super) struct Attempt<R: Runtime<DType = DType>> {
    pub end: f64,
    pub dt: f64,
    pub stage: SurfaceCouplingStage<R>,
    /// Reserved donor water at every group, `[S + C + P]`.
    pub available: Tensor<R>,
    /// Surface-to-channel inflow velocity along each link's reach, `[L]`.
    pub surface_velocity: Tensor<R>,
    /// Channel-to-surface inflow velocity per link, `[L]` per axis.
    pub channel_velocity: [Tensor<R>; 2],
    /// Port fluxes and ported reach reservations of this attempt.
    pub ports: PortBinding<R>,
    /// Pool operands of this attempt, or `None` without pools.
    pub pools: Option<PoolAttempt<R>>,
}

/// Kept child trials of one attempt.
pub(super) type Trials<R> = TrialMemo<Solver<R>, ChannelReach<R>, LevelPoolBatch<R>>;

/// Complete candidate children and exchange of one map evaluation.
pub(super) struct Evaluation<R: Runtime<DType = DType>> {
    pub surface: Solver<R>,
    /// Advanced linked reaches in `Context::linked` order.
    pub linked: Vec<ChannelReach<R>>,
    /// Candidate pools, or `None` without pools or above a top curve knot.
    pub pools: Option<LevelPoolBatch<R>>,
    /// The `pool-capacity-exceeded` rejection of a candidate above a top curve knot. The packet
    /// then holds saturated pool stages.
    pub pool_overflow: Option<String>,
    pub exchange: Exchange<R>,
    pub regime: Tensor<R>,
    /// Midpoint stage at every group, `[S + C + P]`.
    pub midpoint: Tensor<R>,
    /// Output iterate: end stages, signed volumes, end support, regimes.
    pub packet: Tensor<R>,
    /// Change one rounding of the midpoint heads can cause: in each link volume, `[L]`, and in
    /// each end stage, `[S + C + P]`. See [`Context::resolution`].
    pub resolution: [Tensor<R>; 2],
}

/// Candidate children of one completion.
pub(super) struct Completed<R: Runtime<DType = DType>> {
    pub surface: Solver<R>,
    pub linked: Vec<ChannelReach<R>>,
    /// Pool completion, or `None` without pools.
    pub pools: Option<PoolCompletion<R>>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Builds the operands of the attempt ending at `stage.end()`.
    ///
    /// Channel slots reserve native face outflow over the duration and caller withdrawals.
    /// A ported reach's face outflow includes its port export. Pools offer their post-source
    /// water. `pools` holds the accepted pool child and the accepted-step count after the attempt.
    pub fn attempt(
        &self,
        start: &Start<R>,
        stage: SurfaceCouplingStage<R>,
        ports: PortBinding<R>,
        pools: Option<(&PoolChild<R>, usize)>,
    ) -> Result<Attempt<R>, SurfaceChannelError> {
        let c = &self.client;
        let dt = stage.dt();
        let slots = match &start.slots {
            Some(slots) => slots.clone(),
            None => self.slot_operands(|reach| ports.reservation(start, reach))?,
        };
        let mut withdrawal = vec![0.0; self.slot_count];
        for reach in &start.forcing.reaches {
            for item in &reach.withdrawals {
                let volume = if item.amount.is_rate() {
                    item.amount.scalar() * dt
                } else {
                    item.amount.scalar()
                };
                if let Some(slot) = withdrawal.get_mut(item.slot) {
                    *slot += volume;
                }
            }
        }
        let margins: Vec<f64> = self
            .slot_withdrawals
            .iter()
            .map(|&count| SLOT_MARGIN_ULPS * f64::EPSILON * (count as f64 + SLOT_MARGIN_BASE))
            .collect();
        let device = c.device();
        let margin = Tensor::<R>::from_slice(&margins, &[margins.len()], device)?;
        let withdrawal = Tensor::<R>::from_slice(&withdrawal, &[withdrawal.len()], device)?;
        let kept = c.mul(&slots.donor, &c.sub(&self.scalar(1.0)?, &margin)?)?;
        let reserved = c.mul(
            &c.add(&c.mul_scalar(&slots.outgoing, dt)?, &withdrawal)?,
            &c.add_scalar(&margin, 1.0)?,
        )?;
        let remaining = c.maximum(&c.sub(&kept, &reserved)?, &self.scalar(0.0)?)?;
        let pools = match (pools, &start.pools) {
            (Some((child, steps)), Some(pool_start)) => {
                Some(self.pool_attempt(child, pool_start, stage.end(), steps)?)
            }
            (None, None) => None,
            _ => {
                return Err(SurfaceChannelError::Numerical(
                    "an attempt lacks its pool child or pool forcing".into(),
                ));
            }
        };
        let available = match &pools {
            Some(pools) => c.cat(
                &[stage.available_volume_m3(), &remaining, &pools.available],
                0,
            )?,
            None => c.cat(&[stage.available_volume_m3(), &remaining], 0)?,
        };
        let [ux, uy] = stage.velocity_xy();
        let zero = Tensor::<R>::zeros(&[1], DType::F64, device)?;
        let lane = |values: &Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
            Ok(c.index_select(&c.cat(&[values, &zero], 0)?, 0, &self.lane_index)?)
        };
        let surface_velocity = c.add(
            &c.mul(&lane(ux)?, &self.tangent[0])?,
            &c.mul(&lane(uy)?, &self.tangent[1])?,
        )?;
        Ok(Attempt {
            end: stage.end(),
            dt,
            stage,
            available,
            surface_velocity,
            channel_velocity: slots.channel_velocity,
            ports,
            pools,
        })
    }

    /// Evaluates the map at iterate `x` from the `accepted` children.
    ///
    /// Link volumes pass the levelling cap of `levelling`, then the donor limit.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        accepted: Accepted<'_, R>,
        reach_ids: &[&str],
        start: &Start<R>,
        attempt: &Attempt<R>,
        levelling: &Levelling<R>,
        x: &Tensor<R>,
        trials: &Trials<R>,
    ) -> Result<Evaluation<R>, SurfaceChannelError> {
        let c = &self.client;
        let n = self.group_count();
        let l = self.link_count;
        if x.shape() != [2 * (n + l)] {
            return Err(SurfaceChannelError::Numerical(format!(
                "iterate shape={:?}, expected [{}]: supply the coordinator packet",
                x.shape(),
                2 * (n + l)
            )));
        }
        let estimate = x.narrow(0, 0, n)?;
        let midpoint = c.midpoint(&start.stage, &estimate)?;
        let end_wet = x.narrow(0, n + l, n)?;
        self.check_pool_midpoints(&start.stage, &estimate, &start.wet, &end_wet)?;
        let wet = c.maximum(&start.wet, &end_wet)?;
        let law = self.weir(&midpoint, &wet, &levelling.level)?;
        let proposed = self
            .link_structures()?
            .held_volumes(&law, attempt.dt)
            .map_err(|error| SurfaceChannelError::Numerical(format!("link volumes: {error}")))?;
        let level = self.level(&proposed, levelling)?;
        let mut exchange = self.exchange(proposed, level, &attempt.available)?;
        let completed = self.complete(accepted, reach_ids, start, attempt, &exchange, trials)?;
        let borrowed: Vec<&ChannelReach<R>> = completed.linked.iter().collect();
        let (mut end_stage, mut end_wet) = self.stages(&completed.surface, &borrowed, reach_ids)?;
        let (pools, pool_overflow) = match completed.pools {
            Some(completion) => {
                exchange.aggregation_bound =
                    charge(exchange.aggregation_bound, completion.bound_m3)?;
                let (stage, wet, message) = match (&completion.batch, completion.overflow) {
                    (Some(batch), _) => {
                        let (stage, wet) = self.pool_stages(batch)?;
                        (stage, wet, None)
                    }
                    (None, Some(overflow)) => {
                        (overflow.stage, overflow.wet, Some(overflow.message))
                    }
                    (None, None) => {
                        return Err(SurfaceChannelError::Numerical(
                            "a pool completion lacks its candidate and its overflow".into(),
                        ));
                    }
                };
                end_stage = c.cat(&[&end_stage, &stage], 0)?;
                end_wet = c.cat(&[&end_wet, &wet], 0)?;
                (completion.batch, message)
            }
            None => (None, None),
        };
        let packet = c.cat(&[&end_stage, &exchange.signed, &end_wet, &law.regime], 0)?;
        let resolution = self.resolution(&law, &exchange.signed, &midpoint, levelling)?;
        Ok(Evaluation {
            surface: completed.surface,
            linked: completed.linked,
            pools,
            pool_overflow,
            exchange,
            regime: law.regime,
            midpoint,
            packet,
            resolution,
        })
    }

    /// Completes the surface, every linked reach, and the pools with the limited volumes of
    /// `exchange`.
    ///
    /// `accepted` without a pool child leaves the pools out. A child whose operands repeat a trial
    /// kept in `trials` takes the kept result.
    pub fn complete(
        &self,
        accepted: Accepted<'_, R>,
        reach_ids: &[&str],
        start: &Start<R>,
        attempt: &Attempt<R>,
        exchange: &Exchange<R>,
        trials: &Trials<R>,
    ) -> Result<Completed<R>, SurfaceChannelError> {
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let to_b = c.where_cond(&c.gt(&exchange.signed, &zero)?, &exchange.signed, &zero)?;
        let to_a = c.where_cond(
            &c.lt(&exchange.signed, &zero)?,
            &c.neg(&exchange.signed)?,
            &zero,
        )?;
        let s = self.surface_count;
        let packed = if s == 0 {
            c.cat(&[&to_b, &to_a], 0)?
        } else {
            c.cat(
                &[
                    &to_b,
                    &to_a,
                    &exchange.incoming.narrow(0, 0, s)?,
                    &exchange.outgoing.narrow(0, 0, s)?,
                ],
                0,
            )?
        };
        let reach_links: Vec<&[usize]> = self.linked.iter().map(|e| e.links.as_slice()).collect();
        let keys = TrialKeys::read(&packed, self.link_count, &reach_links)?;
        let next_surface = trials.surface(&keys.surface, || {
            self.complete_surface(accepted.surface, attempt, exchange, &to_a)
        })?;
        let mut linked = Vec::with_capacity(self.linked.len());
        for ((index, entry), key) in self.linked.iter().enumerate().zip(&keys.linked) {
            let reach = &accepted.reaches[entry.reach];
            let next = trials.reach(index, key, || {
                let bound = &start.forcing.reaches[entry.reach];
                let mut inputs = bound.lateral.clone();
                inputs.extend(self.exchange_inputs(
                    index,
                    &to_b,
                    &to_a,
                    &attempt.surface_velocity,
                )?);
                let channel = |source| SurfaceChannelError::Channel {
                    reach_id: reach_ids[entry.reach].to_string(),
                    source,
                };
                let reservation = attempt.ports.reservation(start, entry.reach)?;
                let forcing = reach
                    .bind_coupling_reserved(reservation, attempt.end, &inputs)
                    .map_err(channel)?;
                advanced(
                    reach.trial_coupled(attempt.end, &forcing, None, None),
                    start.time,
                    attempt.end,
                    reach_ids[entry.reach],
                )
            })?;
            linked.push(next);
        }
        let pools = match (accepted.pools, &start.pools, &attempt.pools) {
            (Some(child), Some(pool_start), Some(pool_attempt)) => {
                let offset = s + self.slot_count;
                let p = self.pool_count;
                Some(self.complete_pools(
                    (child, pool_start, pool_attempt),
                    &exchange.incoming.narrow(0, offset, p)?,
                    &exchange.outgoing.narrow(0, offset, p)?,
                    trials,
                )?)
            }
            (None, _, _) => None,
            _ => {
                return Err(SurfaceChannelError::Numerical(
                    "a completion lacks its pool child, pool forcing, or pool attempt".into(),
                ));
            }
        };
        Ok(Completed {
            surface: next_surface,
            linked,
            pools,
        })
    }

    /// Completes the surface interval with the lane totals of the limited link volumes.
    ///
    /// `to_a` holds each link's volume toward endpoint A. Only bank links, whose endpoint A is a
    /// surface lane, carry velocity into the surface.
    fn complete_surface(
        &self,
        surface: &Solver<R>,
        attempt: &Attempt<R>,
        exchange: &Exchange<R>,
        to_a: &Tensor<R>,
    ) -> Result<Solver<R>, SurfaceChannelError> {
        let Some(links) = &self.links else {
            return self.complete_alone(surface, &attempt.stage);
        };
        let c = &self.client;
        let s = self.surface_count;
        let inflow = exchange.incoming.narrow(0, 0, s)?;
        let withdrawal = exchange.outgoing.narrow(0, 0, s)?;
        let zero = self.scalar(0.0)?;
        let has = c.gt(&inflow, &zero)?;
        let safe = c.where_cond(&has, &inflow, &self.scalar(1.0)?)?;
        let velocity = |axis: usize| -> Result<Tensor<R>, SurfaceChannelError> {
            let momentum = c.mul(to_a, &attempt.channel_velocity[axis])?;
            let lane = self
                .lanes
                .aggregate(c, &momentum, AggregationKind::Momentum)
                .map_err(transfer_error)?;
            Ok(c.where_cond(&has, &c.div(&lane.value, &safe)?, &zero)?)
        };
        let source = ExchangeSource::new(
            links,
            inflow.contiguous()?,
            withdrawal.contiguous()?,
            [velocity(0)?, velocity(1)?],
        )?;
        match surface.coupling_complete(&attempt.stage, Some(&source))? {
            CouplingCompletion::Accepted(next) => Ok(*next),
            CouplingCompletion::Retry(reason) => Err(SurfaceChannelError::Numerical(format!(
                "surface completion: {reason}. Shorten the common interval"
            ))),
        }
    }
}

/// Accepted reach of a coupled trial, or a retryable shortening request.
pub(super) fn advanced<R: Runtime<DType = DType>>(
    trial: Result<ChannelCouplingTrial<R>, crate::channel::ChannelError>,
    start: f64,
    end: f64,
    reach_id: &str,
) -> Result<ChannelReach<R>, SurfaceChannelError> {
    match trial {
        Ok(ChannelCouplingTrial::Accepted(next)) => Ok(*next),
        Ok(ChannelCouplingTrial::Shorter {
            suggested_end_s,
            reason,
        }) => Err(SurfaceChannelError::RecomputeInterval {
            start_s: start,
            requested_end_s: end,
            suggested_end_s,
            reason: format!("reach={reach_id:?}: {reason}"),
        }),
        Err(source) => Err(SurfaceChannelError::Channel {
            reach_id: reach_id.to_string(),
            source,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::structures::scenario::{
        DT, Tanks, TestResult, check, inflow, orifice_law, receipt, step, structure, withdrawal,
    };
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    /// Orifice crown of both cases, in m.
    const CROWN: f64 = 0.3;

    fn culvert() -> TestResult<super::super::structures::SurfaceChannelStructure> {
        Ok(structure(
            "culvert",
            orifice_law(0.1, CROWN)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            [0.0, 0.0],
        ))
    }

    /// Caller inflow raises the reach above the surface, then caller withdrawal lowers it.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Some interval of the inflow phase carries water to the surface (negative volume).
    /// - Some later interval of the withdrawal phase carries water to the reach (positive).
    /// - Every interval passes its mass bound, both pairings, and the composite balance.
    #[test]
    fn caller_water_reverses_the_orifice() -> TestResult {
        let mut batch = Tanks::at(0.7, 0.7).build(vec![culvert()?])?;
        let mut to_surface = false;
        for _ in 0..20 {
            batch = step(&batch, DT, &[inflow(0.02)])?;
            check(&batch)?;
            to_surface |= receipt(&batch, "culvert")?.signed < 0.0;
        }
        let mut to_reach = false;
        for _ in 0..40 {
            batch = step(&batch, DT, &[withdrawal(0.02)])?;
            check(&batch)?;
            to_reach |= receipt(&batch, "culvert")?.signed > 0.0;
        }
        assert!(to_surface && to_reach, "{to_surface} {to_reach}");
        Ok(())
    }

    /// A caller withdrawal would draw the reach below the orifice crown over the full interval.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The trial accepts after at least one retry, short of the requested end.
    /// - Both midpoint heads stand above the crown.
    /// - The interval passes its mass bound, both pairings, and the composite balance.
    #[test]
    fn a_crown_crossing_retries_a_shorter_interval() -> TestResult {
        let batch = Tanks::at(0.36, 0.35).build(vec![culvert()?])?;
        let batch = step(&batch, DT, &[withdrawal(1.0)])?;
        check(&batch)?;
        let interval = batch.last_step()?.ok_or("an accepted interval")?;
        assert!(interval.retries >= 1, "{interval:?}");
        assert!(interval.end_s < DT, "{interval:?}");
        let link = receipt(&batch, "culvert")?;
        assert!(link.midpoint.iter().all(|&head| head > CROWN), "{link:?}");
        Ok(())
    }
}
