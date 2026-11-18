//! Child completions of one evaluation: the surface, every linked reach, every pool-ended reach,
//! and the pools.
//!
//! Each linked reach takes its links' limited volumes on its synthesized bindings. A pool-ended
//! reach binds through its reservation under the evaluation's reservoir heads. Its reach end
//! volume then enters its pool unscaled, beside the pool's link volumes.

use super::SurfaceChannelError;
use super::context::{Context, Ops, transfer_error};
use super::evaluate::{Accepted, Attempt, Trials};
use super::exchange::Exchange;
use super::memo::{TrialKey, TrialKeys};
use super::pool_ends::EndVolumes;
use super::pools::{PoolCompletion, charge};
use super::reserve::Reserved;
use super::start::Start;
use crate::channel::{ChannelReach, ChannelTensorLateralInput};
use crate::numerics::water_transfers::AggregationKind;
use crate::solver::Solver;
use hydr_channel::internal::ChannelCouplingTrial;
use hydr_solver::internal::{CouplingCompletion, ExchangeSource};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Candidate children of one completion.
pub(super) struct Completed<R: Runtime<DType = DType>> {
    pub surface: Solver<R>,
    pub linked: Vec<ChannelReach<R>>,
    /// Pool-ended reaches without links, in `PoolEndTopology::ended` order. Empty without a pool
    /// child.
    pub ended: Vec<ChannelReach<R>>,
    /// Pool-end volumes, or `None` without pool ends or without a pool child.
    pub ends: Option<EndVolumes<R>>,
    /// Pool completion, or `None` without pools.
    pub pools: Option<PoolCompletion<R>>,
}

/// Start, attempt, and evaluation operands of one completion.
pub(super) type Operands<'a, R> = (&'a Start<R>, &'a Attempt<R>, &'a Reserved<R>);

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Completes the surface, every linked reach, every pool-ended reach, and the pools with the
    /// limited volumes of `exchange`.
    ///
    /// `accepted` without a pool child leaves the pools and the pool-ended reaches without links
    /// out. A child whose operands repeat a trial kept in `trials` takes the kept result. A
    /// pool-ended reach's key adds the bits of its reservoir heads.
    pub fn complete(
        &self,
        accepted: Accepted<'_, R>,
        reach_ids: &[&str],
        (start, attempt, reserved): Operands<'_, R>,
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
        // A linked pool-ended reach sets the channel velocity of each evaluation from its
        // reservoir heads, so the surface key holds them too.
        let surface_key = match (&attempt.channel_velocity, &reserved.ends) {
            (None, Some(binding)) => keys.surface.joined(&binding.head_bits),
            _ => keys.surface.clone(),
        };
        let next_surface = trials.surface(&surface_key, || {
            self.complete_surface(accepted.surface, attempt, reserved, exchange, &to_a)
        })?;
        let head_bits = |reach: usize| -> Option<Vec<u64>> {
            let topology = self.pool_ends.as_ref()?;
            let binding = reserved.ends.as_ref()?;
            topology
                .owns(reach)
                .then(|| binding.reach_bits(topology, reach))
        };
        let complete_reach = |reach: usize,
                              inputs: &[ChannelTensorLateralInput<R>]|
         -> Result<ChannelReach<R>, SurfaceChannelError> {
            let child = &accepted.reaches[reach];
            let reservation = reserved.reservation(start, attempt, reach)?;
            let forcing = child
                .bind_coupling_reserved(reservation, attempt.end, inputs)
                .map_err(|source| SurfaceChannelError::Channel {
                    reach_id: reach_ids[reach].to_string(),
                    source,
                })?;
            advanced(
                child.trial_coupled(attempt.end, &forcing, None, None),
                start.time,
                attempt.end,
                reach_ids[reach],
            )
        };
        let mut linked = Vec::with_capacity(self.linked.len());
        for ((index, entry), key) in self.linked.iter().enumerate().zip(&keys.linked) {
            let key = match head_bits(entry.reach) {
                Some(bits) => key.joined(&bits),
                None => key.clone(),
            };
            let next = trials.reach(index, &key, || {
                let bound = &start.forcing.reaches[entry.reach];
                let mut inputs = bound.lateral.clone();
                inputs.extend(self.exchange_inputs(
                    index,
                    &to_b,
                    &to_a,
                    &attempt.surface_velocity,
                )?);
                complete_reach(entry.reach, &inputs)
            })?;
            linked.push(next);
        }
        let mut ended = Vec::new();
        let ends = match (accepted.pools, &self.pool_ends) {
            (Some(_), Some(topology)) => {
                for (offset, &reach) in topology.ended.iter().enumerate() {
                    let key = TrialKey::from_bits(head_bits(reach).unwrap_or_default());
                    let next = trials.reach(self.linked.len() + offset, &key, || {
                        complete_reach(reach, &start.forcing.reaches[reach].lateral)
                    })?;
                    ended.push(next);
                }
                Some(self.end_volumes(&linked, &ended, reach_ids)?)
            }
            _ => None,
        };
        let pools = match (accepted.pools, &start.pools, &attempt.pools) {
            (Some(child), Some(pool_start), Some(pool_attempt)) => {
                let offset = s + self.slot_count;
                let p = self.pool_count;
                let incoming = exchange.incoming.narrow(0, offset, p)?;
                let outgoing = exchange.outgoing.narrow(0, offset, p)?;
                let (incoming, outgoing) = match &ends {
                    Some(ends) => (
                        c.add(&incoming, &ends.incoming)?,
                        c.add(&outgoing, &ends.outgoing)?,
                    ),
                    None => (incoming, outgoing),
                };
                let mut completion = self.complete_pools(
                    (child, pool_start, pool_attempt),
                    &incoming,
                    &outgoing,
                    trials,
                )?;
                if let Some(ends) = &ends {
                    completion.bound_m3 = charge(completion.bound_m3, ends.bound_m3)?;
                }
                Some(completion)
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
            ended,
            ends,
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
        reserved: &Reserved<R>,
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
            let momentum = c.mul(to_a, &reserved.channel_velocity[axis])?;
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
