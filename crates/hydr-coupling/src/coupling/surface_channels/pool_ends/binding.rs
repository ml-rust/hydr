//! Per-evaluation pool-end operands: reservoir heads, pool-ended reach reservations, and the
//! end volumes of the completed reaches.
//!
//! Each evaluation reads its pools' midpoint stages from the iterate. Every pool-ended reach then
//! binds `QuiescentReservoir` at those heads and reserves its accepted-start operands under them.
//! A ported end of the same reach keeps its flux port, and every other end its caller policy.

use super::PoolEndTopology;
use crate::channel::ChannelError;
use crate::channel::{ChannelReach, ChannelTensorBoundary};
use crate::coupling::links::LinkError;
use crate::coupling::links::pool_ends::{self, EndHeads, ReachEnds};
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::{Context, Ops, transfer_error};
use crate::coupling::surface_channels::evaluate::Attempt;
use crate::coupling::surface_channels::pools::charge;
use crate::coupling::surface_channels::start::Start;
use crate::coupling::surface_channels::trial::LIMIT_SAFETY;
use crate::numerics::water_transfers::AggregationKind;
use hydr_channel::internal::ChannelCouplingReservation;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ShapeOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Reservoir heads and pool-ended reach reservations of one evaluation.
pub(in super::super) struct EndBinding<R: Runtime<DType = DType>> {
    /// Reservoir head of every end: its pool's temporal midpoint stage, `[E]`.
    pub heads: Tensor<R>,
    /// Bits of `heads`, in pool-end order.
    pub head_bits: Vec<u64>,
    /// Per setup reach: the reservation of a pool-ended reach under its reservoir heads.
    reservations: Vec<Option<ChannelCouplingReservation<R>>>,
}

impl<R: Runtime<DType = DType>> EndBinding<R> {
    /// The reservation of pool-ended setup reach `reach`, if any.
    pub fn reservation(&self, reach: usize) -> Option<&ChannelCouplingReservation<R>> {
        self.reservations.get(reach).and_then(Option::as_ref)
    }

    /// Head bits of the pool ends of setup reach `reach`, left end first.
    pub fn reach_bits(&self, topology: &PoolEndTopology<R>, reach: usize) -> Vec<u64> {
        topology
            .reach_ends
            .get(reach)
            .map_or(&[None, None], |ends| ends)
            .iter()
            .flatten()
            .filter_map(|&end| self.head_bits.get(end).copied())
            .collect()
    }
}

/// End volumes of one completion with their pool totals.
pub(in super::super) struct EndVolumes<R: Runtime<DType = DType>> {
    /// Signed volume of every end, positive from the reach into the pool, `[E]`.
    pub volume: Tensor<R>,
    /// Water every pool takes from its ends, `[P]`.
    pub incoming: Tensor<R>,
    /// Water every pool gives its ends, `[P]`.
    pub outgoing: Tensor<R>,
    /// Summed aggregation rounding bound of both totals, in m³.
    pub bound_m3: f64,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Binds every pool-ended reach at the reservoir heads of iterate `x`, or `None` without
    /// pool ends.
    ///
    /// A reach whose start CFL or face depletion limit falls below the attempt duration requests
    /// `0.95` of that limit. A channel rejection of the reservoir trace names `pool-end-regime`
    /// and stays retryable.
    pub fn bind_ends(
        &self,
        start: &Start<R>,
        attempt: &Attempt<R>,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
        x: &Tensor<R>,
    ) -> Result<Option<EndBinding<R>>, SurfaceChannelError> {
        let Some(topology) = &self.pool_ends else {
            return Ok(None);
        };
        let c = &self.client;
        let offset = self.surface_count + self.slot_count;
        let p = self.pool_count;
        let EndHeads { heads, .. } = pool_ends::end_heads(
            c,
            &start.stage.narrow(0, offset, p)?,
            &x.narrow(0, offset, p)?,
            &topology.pools,
        )
        .map_err(LinkError::surface_channel)?;
        let head_bits = heads
            .contiguous()?
            .try_to_vec::<f64>()?
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let mut reservations: Vec<Option<ChannelCouplingReservation<R>>> =
            (0..reaches.len()).map(|_| None).collect();
        let zeros = Tensor::<R>::zeros(&[self.link_count], DType::F64, c.device())?;
        for &reach in &self.reach_order {
            let Some(&owned) = topology.reach_ends.get(reach) else {
                continue;
            };
            if owned.iter().all(Option::is_none) {
                continue;
            }
            let reach_id = reach_ids[reach];
            let channel = |source| SurfaceChannelError::Channel {
                reach_id: reach_id.to_string(),
                source,
            };
            let operands = ReachEnds::new(&heads, &topology.losses, owned)
                .map_err(LinkError::surface_channel)?;
            let bound = &start.forcing.reaches[reach];
            let ports = self
                .ports
                .as_ref()
                .and_then(|ports| ports.reach_ports.get(reach));
            let mut policies = Vec::with_capacity(2);
            for side in 0..2 {
                let policy = match operands.operands(side) {
                    Some(reservoir) => ChannelTensorBoundary::QuiescentReservoir {
                        elevation_m: reservoir.elevation_m.clone(),
                        entrance_loss_coefficient: reservoir.entrance_loss_coefficient.clone(),
                    },
                    None => {
                        let port = ports
                            .and_then(|ends| ends[side])
                            .and_then(|port| attempt.ports.boundary(port));
                        let caller = if side == 0 { &bound.left } else { &bound.right };
                        port.or(caller.as_ref()).cloned().ok_or_else(|| {
                            SurfaceChannelError::Numerical(
                                "a pool-ended reach end lacks its caller policy and its port"
                                    .into(),
                            )
                        })?
                    }
                };
                policies.push(policy);
            }
            let inputs = self.reservation_inputs(reach, &bound.lateral, &zeros)?;
            let child = &reaches[reach];
            let forcing = child
                .bind_coupling_tensor(
                    start.time,
                    attempt.end,
                    policies.first(),
                    policies.get(1),
                    &inputs,
                )
                .map_err(channel)?;
            let reservation = child.coupling_reservation(&forcing).map_err(|source| {
                match source {
                    ChannelError::Numerical { .. } => SurfaceChannelError::Numerical(format!(
                        "pool-end-regime: reach={reach_id:?} rejects its reservoir trace: {source}. A pool end needs wet subcritical water at both sides; shorten the common interval"
                    )),
                    source => channel(source),
                }
            })?;
            let limit = reservation
                .start_cfl_s
                .min(reservation.face_depletion_limit_s);
            if attempt.dt > limit {
                return Err(SurfaceChannelError::RecomputeInterval {
                    start_s: start.time,
                    requested_end_s: attempt.end,
                    suggested_end_s: start.time + LIMIT_SAFETY * limit,
                    reason: format!(
                        "pool-ended reach={reach_id:?} start CFL={}, face depletion={}",
                        reservation.start_cfl_s, reservation.face_depletion_limit_s
                    ),
                });
            }
            reservations[reach] = Some(reservation);
        }
        Ok(Some(EndBinding {
            heads,
            head_bits,
            reservations,
        }))
    }

    /// End volumes of every pool end from the completed reaches, with their pool totals.
    ///
    /// `linked` holds the completed linked reaches in `Context::linked` order and `ended` the
    /// completed pool-ended reaches without links, in `PoolEndTopology::ended` order.
    pub fn end_volumes(
        &self,
        linked: &[ChannelReach<R>],
        ended: &[ChannelReach<R>],
        reach_ids: &[&str],
    ) -> Result<EndVolumes<R>, SurfaceChannelError> {
        let topology = self.end_topology()?;
        let c = &self.client;
        let mut parts = Vec::with_capacity(topology.ends.len());
        for end in &topology.ends {
            let reach = end.reach_index;
            let completed = match self.linked.iter().position(|entry| entry.reach == reach) {
                Some(index) => linked.get(index),
                None => topology
                    .ended
                    .iter()
                    .position(|&r| r == reach)
                    .and_then(|index| ended.get(index)),
            }
            .ok_or_else(|| {
                SurfaceChannelError::Numerical("a pool end lacks its completed reach".into())
            })?;
            let endpoints =
                completed
                    .coupling_endpoints()
                    .map_err(|source| SurfaceChannelError::Channel {
                        reach_id: reach_ids[reach].to_string(),
                        source,
                    })?;
            parts.push(endpoints.outward_volume_m3.narrow(0, end.side, 1)?);
        }
        let volume = c.cat(&parts.iter().collect::<Vec<_>>(), 0)?;
        let zero = self.scalar(0.0)?;
        let incoming = topology
            .tree
            .aggregate(c, &c.maximum(&volume, &zero)?, AggregationKind::Water)
            .map_err(transfer_error)?;
        let outgoing = topology
            .tree
            .aggregate(
                c,
                &c.maximum(&c.neg(&volume)?, &zero)?,
                AggregationKind::Water,
            )
            .map_err(transfer_error)?;
        let bound_m3 = c
            .cat(&[&incoming.bound, &outgoing.bound], 0)?
            .try_to_vec::<f64>()?
            .into_iter()
            .try_fold(0.0, charge)?;
        Ok(EndVolumes {
            volume,
            incoming: incoming.value,
            outgoing: outgoing.value,
            bound_m3,
        })
    }
}
