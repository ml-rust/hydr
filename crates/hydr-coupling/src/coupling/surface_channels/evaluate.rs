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
//!
//! A pool-ended reach reserves under the reservoir heads of each iterate, so its slots' donor
//! water is a function of the iterate. Its end volume joins the packet beside the link volumes.

use super::SurfaceChannelError;
use super::context::{Context, Ops};
use super::exchange::Exchange;
use super::levelling::Levelling;
use super::link_law::LawAt;
use super::memo::TrialMemo;
use super::pools::{PoolAttempt, PoolChild, charge};
use super::ports::PortBinding;
use super::runoff::RunoffAttempt;
use super::start::Start;
use super::structures::LinkOperations;
use crate::channel::{ChannelReach, ChannelTensorLateralInput};
use crate::coupling::links::culvert_roots::CulvertRoots;
use crate::solver::Solver;
use crate::storage::LevelPoolBatch;
use hydr_solver::internal::SurfaceCouplingStage;
use numr::dtype::DType;
use numr::ops::{BinaryOps, IndexingOps, MidpointOps, ShapeOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

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
    /// Reserved donor water at every group, `[S + C + P]`, or `None` when a linked reach is
    /// pool-ended. Each evaluation then reserves it under its reservoir heads.
    pub available: Option<Tensor<R>>,
    /// Surface-to-channel inflow velocity along each link's reach, `[L]`.
    pub surface_velocity: Tensor<R>,
    /// Channel-to-surface inflow velocity per link, `[L]` per axis, or `None` when a linked
    /// reach is pool-ended.
    pub channel_velocity: Option<[Tensor<R>; 2]>,
    /// Caller withdrawal volume and margin of every slot, each `[C]`.
    pub slot_reserve: [Tensor<R>; 2],
    /// Port fluxes and ported reach reservations of this attempt.
    pub ports: PortBinding<R>,
    /// Pool operands of this attempt, or `None` without pools.
    pub pools: Option<PoolAttempt<R>>,
    /// Link laws under the operations of this attempt.
    ///
    /// They depend on no iterate, so every evaluation of the attempt takes them.
    pub operations: Arc<LinkOperations<R>>,
    /// Culvert roots of the attempt's latest law evaluation.
    pub culvert_roots: CulvertRoots<R>,
    /// Runoff inflow inputs of every setup reach, fixed for the attempt. Empty without runoff.
    pub runoff_laterals: Vec<Vec<ChannelTensorLateralInput<R>>>,
    /// Runoff inflow of every surface exchange lane, `[S + R]`, fixed for the attempt. `None`
    /// without surface runoff.
    pub surface_runoff: Option<Tensor<R>>,
}

/// Kept child trials of one attempt.
pub(super) type Trials<R> = TrialMemo<Solver<R>, ChannelReach<R>, LevelPoolBatch<R>>;

/// Complete candidate children and exchange of one map evaluation.
pub(super) struct Evaluation<R: Runtime<DType = DType>> {
    pub surface: Solver<R>,
    /// Advanced linked reaches in `Context::linked` order.
    pub linked: Vec<ChannelReach<R>>,
    /// Advanced pool-ended reaches without links, in `PoolEndTopology::ended` order.
    pub ended: Vec<ChannelReach<R>>,
    /// Pool-end volumes, positive into the pools, `[E]`, or `None` without pool ends.
    pub end_volume: Option<Tensor<R>>,
    /// Reservoir head of every pool end, `[E]`, or `None` without pool ends.
    pub end_heads: Option<Tensor<R>>,
    /// Wet support of every pool end at the candidate end, `[E]`, or `None` without pool ends.
    pub end_support: Option<Tensor<R>>,
    /// Candidate pools, or `None` without pools or above a top curve knot.
    pub pools: Option<LevelPoolBatch<R>>,
    /// The `pool-capacity-exceeded` rejection of a candidate above a top curve knot. The packet
    /// then holds saturated pool stages.
    pub pool_overflow: Option<String>,
    pub exchange: Exchange<R>,
    /// Regime code of every link, `[L]`.
    pub regime: Tensor<R>,
    /// Midpoint stage at every group, `[S + C + P]`.
    pub midpoint: Tensor<R>,
    /// Output iterate: end stages `[S + C + P]`, signed link and pool-end volumes `[L + E]`, end
    /// wet masks `[S + C + P]`, link regimes `[L]`, and pool-end support `[E]`.
    pub packet: Tensor<R>,
    /// Change one rounding of the midpoint heads can cause: in each signed volume, `[L + E]`,
    /// and in each end stage, `[S + C + P]`. See [`Context::resolution`]. Pool-end volumes take
    /// zero: the reach sets them.
    pub resolution: [Tensor<R>; 2],
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
    /// `operations` holds the link laws under the attempt's operations. `runoff` holds the
    /// attempt's runoff delivery, which pools and reaches receive.
    pub fn attempt(
        &self,
        start: &Start<R>,
        (stage, operations): (SurfaceCouplingStage<R>, Arc<LinkOperations<R>>),
        ports: PortBinding<R>,
        pools: Option<(&PoolChild<R>, usize)>,
        runoff: Option<&RunoffAttempt<R>>,
    ) -> Result<Attempt<R>, SurfaceChannelError> {
        let c = &self.client;
        let dt = stage.dt();
        let pool_ended = self
            .linked
            .iter()
            .any(|entry| self.is_pool_ended(entry.reach));
        let slots = match &start.slots {
            Some(slots) => Some(slots.clone()),
            None if pool_ended => None,
            None => Some(self.slot_operands(|reach| ports.reservation(start, reach))?),
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
        let slot_reserve = [withdrawal, margin];
        let pools = match (pools, &start.pools) {
            (Some((child, steps)), Some(pool_start)) => {
                let native = runoff.and_then(|runoff| runoff.pool_volumes.as_ref());
                Some(self.pool_attempt((child, pool_start), stage.end(), steps, native)?)
            }
            (None, None) => None,
            _ => {
                return Err(SurfaceChannelError::Numerical(
                    "an attempt lacks its pool child or pool forcing".into(),
                ));
            }
        };
        let available = slots
            .as_ref()
            .map(|slots| self.donor_water(&stage, slots, dt, &slot_reserve, pools.as_ref()))
            .transpose()?;
        // The stage covers the exchange lanes. Links read the link lanes only.
        let [ux, uy] = stage.velocity_xy();
        let zero = Tensor::<R>::zeros(&[1], DType::F64, device)?;
        let lane = |values: &Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
            let values = self.link_lanes(values)?;
            Ok(c.index_select(&c.cat(&[&values, &zero], 0)?, 0, &self.lane_index)?)
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
            channel_velocity: slots.map(|slots| slots.channel_velocity),
            slot_reserve,
            ports,
            pools,
            operations,
            culvert_roots: CulvertRoots::new(start.law.as_ref()),
            runoff_laterals: runoff.map_or_else(Vec::new, |runoff| runoff.laterals.clone()),
            surface_runoff: runoff.and_then(|runoff| runoff.surface.clone()),
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
        let m = self.transfer_count();
        if x.shape() != [2 * (n + m)] {
            return Err(SurfaceChannelError::Numerical(format!(
                "iterate shape={:?}, expected [{}]: supply the coordinator packet",
                x.shape(),
                2 * (n + m)
            )));
        }
        let estimate = x.narrow(0, 0, n)?;
        let midpoint = c.midpoint(&start.stage, &estimate)?;
        let end_wet = x.narrow(0, n + m, n)?;
        self.check_pool_midpoints(&start.stage, &estimate, &start.wet, &end_wet)?;
        let reserved = self.reserve(accepted.reaches, reach_ids, start, attempt, x)?;
        let (law, mut exchange) = if self.link_count == 0 {
            (None, self.no_exchange()?)
        } else {
            // The first iterate holds the accepted stages and wet masks. Its midpoint and wet
            // masks then repeat them in every bit, so it takes the start laws.
            let first = Arc::ptr_eq(&start.operations, &attempt.operations)
                && x.shares_storage_with(&start.initial);
            let law = match (&start.law, first) {
                (Some(law), true) => self.level_weir(law.clone(), &midpoint, &levelling.level)?,
                _ => {
                    let wet = c.maximum(&start.wet, &end_wet)?;
                    self.weir(
                        &attempt.operations,
                        (&midpoint, &wet),
                        &levelling.level,
                        LawAt::Trial(&attempt.culvert_roots),
                    )?
                }
            };
            let proposed = attempt
                .operations
                .structures()?
                .held_volumes(&law, attempt.dt)
                .map_err(|error| {
                    SurfaceChannelError::Numerical(format!("link volumes: {error}"))
                })?;
            let level = self.level(&proposed, levelling)?;
            let exchange = self.exchange(proposed, level, &reserved.available)?;
            (Some(law), exchange)
        };
        let completed = self.complete(
            accepted,
            reach_ids,
            (start, attempt, &reserved),
            &exchange,
            trials,
        )?;
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
        let empty = Tensor::<R>::zeros(&[0], DType::F64, c.device())?;
        let (regime, mut resolution) = match &law {
            Some(law) => (
                law.regime.clone(),
                self.resolution(law, &exchange.signed, &midpoint, levelling)?,
            ),
            None => (
                empty.clone(),
                [empty, Tensor::<R>::zeros(&[n], DType::F64, c.device())?],
            ),
        };
        let (packet, end_volume, end_support) = match &completed.ends {
            Some(ends) => {
                // An overflowing candidate rejects once it converges, so its support stays set.
                let support = match &pools {
                    Some(batch) => self.end_support(batch.volumes_tensor())?,
                    None => Tensor::<R>::ones(ends.volume.shape(), DType::F64, c.device())?,
                };
                let none = Tensor::<R>::zeros(ends.volume.shape(), DType::F64, c.device())?;
                resolution[0] = c.cat(&[&resolution[0], &none], 0)?;
                let packet = c.cat(
                    &[
                        &end_stage,
                        &exchange.signed,
                        &ends.volume,
                        &end_wet,
                        &regime,
                        &support,
                    ],
                    0,
                )?;
                (packet, Some(ends.volume.clone()), Some(support))
            }
            None => (
                c.cat(&[&end_stage, &exchange.signed, &end_wet, &regime], 0)?,
                None,
                None,
            ),
        };
        Ok(Evaluation {
            surface: completed.surface,
            linked: completed.linked,
            ended: completed.ended,
            end_volume,
            end_heads: reserved.ends.map(|ends| ends.heads),
            end_support,
            pools,
            pool_overflow,
            exchange,
            regime,
            midpoint,
            packet,
            resolution,
        })
    }

    /// The exchange of an evaluation without links: empty link tensors and zero group totals.
    fn no_exchange(&self) -> Result<Exchange<R>, SurfaceChannelError> {
        let device = self.client.device();
        let empty = Tensor::<R>::zeros(&[0], DType::F64, device)?;
        let groups = Tensor::<R>::zeros(&[self.group_count()], DType::F64, device)?;
        Ok(Exchange {
            proposed: empty.clone(),
            level: empty.clone(),
            imposed: empty.clone(),
            signed: empty.clone(),
            factor: empty,
            incoming: groups.clone(),
            outgoing: groups,
            aggregation_bound: 0.0,
        })
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
