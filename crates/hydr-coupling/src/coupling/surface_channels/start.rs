//! Accepted-start operands shared by every attempt of one common trial.

use super::SurfaceChannelError;
use super::context::{Context, NodeEnd, Ops};
use super::forcing::BoundForcing;
use super::link_law::LawAt;
use super::pools::{PoolChild, PoolStart};
use super::ports::PortStart;
use super::structures::{LinkLaw, LinkOperations};
use crate::channel::{ChannelReach, ChannelTensorLateralInput, ChannelTensorWaterAmount};
use crate::solver::Solver;
use hydr_channel::internal::ChannelCouplingReservation;
use hydr_structures::internal::TensorStructureResult;
use numr::dtype::DType;
use numr::ops::{BinaryOps, IndexingOps, ShapeOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

/// Accepted surface, reaches in setup order, and the pool child with its bound trial forcing.
pub(super) type StartChildren<'a, R> = (
    &'a Solver<R>,
    &'a [ChannelReach<R>],
    Option<(&'a PoolChild<R>, PoolStart<R>)>,
);

/// Operands that depend only on the accepted state and the caller forcing.
pub(super) struct Start<R: Runtime<DType = DType>> {
    pub time: f64,
    /// Accepted stage at every donor group, `[S + C + P]`.
    pub stage: Tensor<R>,
    /// Accepted wet mask at every donor group, `[S + C + P]`.
    pub wet: Tensor<R>,
    /// First fixed-point iterate: accepted stages, zero volumes, accepted support and regimes.
    pub initial: Tensor<R>,
    /// Link laws at the accepted stages under `operations`, before the equal-heads rule, or
    /// `None` without links.
    ///
    /// The first iterate's midpoint stages and wet masks repeat the accepted ones in every bit,
    /// so its evaluation takes these laws. Their culvert roots seed the root searches of every
    /// attempt. See [`Context::evaluate`].
    pub law: Option<TensorStructureResult<R>>,
    /// Slot operands of every linked reach, or `None` when a linked reach is ported or
    /// pool-ended.
    ///
    /// A ported reach's face outflow depends on its port flux, so each attempt builds them. A
    /// pool-ended reach's face outflow depends on its reservoir heads, so each evaluation does.
    pub slots: Option<SlotOperands<R>>,
    /// Smallest accepted-start CFL or face depletion limit over every reach without a port or a
    /// pool end, in seconds.
    pub channel_limit_s: f64,
    /// Accepted-start reservation of every reach without a port or a pool end, in setup order.
    ///
    /// Every candidate binds through it, so each reach calculates its start fluxes once per trial.
    /// A ported reach reserves once per attempt under its port flux. A pool-ended reach reserves
    /// once per evaluation under its reservoir heads.
    pub reservations: Vec<Option<ChannelCouplingReservation<R>>>,
    pub forcing: BoundForcing<R>,
    /// Ghost states of every port from the accepted reach ends, or `None` without ports.
    pub ports: Option<PortStart<R>>,
    /// Pool forcing of this trial, or `None` without pools.
    pub pools: Option<PoolStart<R>>,
    /// Wet support of every pool end at the accepted start, `[E]`, or `None` without pool ends.
    pub end_support: Option<Tensor<R>>,
    /// Link laws the start regimes and the pump checks used.
    ///
    /// An attempt under other operations takes its own start regimes. See
    /// [`Context::initial_iterate`].
    pub operations: Arc<LinkOperations<R>>,
}

/// Donor operands of every channel slot.
#[derive(Clone)]
pub(super) struct SlotOperands<R: Runtime<DType = DType>> {
    /// Accepted donor water at every slot, `[C]`.
    pub donor: Tensor<R>,
    /// Accepted native face outflow rate at every slot, port ends included, `[C]`.
    pub outgoing: Tensor<R>,
    /// Channel-to-surface inflow velocity per link, `[L]` per axis: start velocity on the tangent.
    pub channel_velocity: [Tensor<R>; 2],
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Exchange inputs for every link of linked reach `linked`, from per-link volume tensors.
    ///
    /// `to_b` holds each link's volume from A to B and `to_a` from B to A. The channel endpoint
    /// takes the volume toward it as inflow and the volume away from it as withdrawal.
    pub fn exchange_inputs(
        &self,
        linked: usize,
        to_b: &Tensor<R>,
        to_a: &Tensor<R>,
        velocity: &Tensor<R>,
    ) -> Result<Vec<ChannelTensorLateralInput<R>>, SurfaceChannelError> {
        let mut inputs = Vec::new();
        for &link in self.linked.get(linked).map_or(&[][..], |r| &r.links) {
            let (inflow, withdrawal) = &self.link_bindings[link];
            let (into, from) = match self.link_ends.get(link) {
                Some([NodeEnd::Channel(_), _]) => (to_a, to_b),
                _ => (to_b, to_a),
            };
            inputs.push(ChannelTensorLateralInput::Inflow {
                boundary_id: inflow.clone(),
                amount: ChannelTensorWaterAmount::IntervalVolume {
                    volume_m3: into.narrow(0, link, 1)?,
                },
                longitudinal_velocity_m_per_s: velocity.narrow(0, link, 1)?,
            });
            inputs.push(ChannelTensorLateralInput::Withdrawal {
                boundary_id: withdrawal.clone(),
                amount: ChannelTensorWaterAmount::IntervalVolume {
                    volume_m3: from.narrow(0, link, 1)?,
                },
            });
        }
        Ok(inputs)
    }

    /// Builds the start operands from the accepted children and the bound forcing.
    ///
    /// `pools` holds the accepted pool child with its bound trial forcing. `operations` holds the
    /// link laws of the first attempt. Its pump operations hold over every attempt of the trial.
    pub fn start(
        &self,
        (surface, reaches, pools): StartChildren<'_, R>,
        reach_ids: &[&str],
        (end, operations): (f64, &Arc<LinkOperations<R>>),
        forcing: BoundForcing<R>,
    ) -> Result<Start<R>, SurfaceChannelError> {
        let (pool_child, pool_start) = match pools {
            Some((child, start)) => (Some(child), Some(start)),
            None => (None, None),
        };
        let c = &self.client;
        let time = surface.time();
        let zeros = Tensor::<R>::zeros(&[self.link_count], DType::F64, c.device())?;
        let mut channel_limit_s = f64::INFINITY;
        let mut reservations: Vec<Option<ChannelCouplingReservation<R>>> =
            (0..reaches.len()).map(|_| None).collect();
        let channel = |reach: usize| {
            move |source| SurfaceChannelError::Channel {
                reach_id: reach_ids[reach].to_string(),
                source,
            }
        };
        let linked_of = |reach: usize| self.linked.iter().position(|r| r.reach == reach);
        for &reach in &self.reach_order {
            if self.is_ported(reach) || self.is_pool_ended(reach) {
                continue;
            }
            let bound = &forcing.reaches[reach];
            let (Some(left), Some(right)) = (&bound.left, &bound.right) else {
                return Err(SurfaceChannelError::Numerical(
                    "an unported reach lacks a caller endpoint policy".into(),
                ));
            };
            let mut inputs = bound.lateral.clone();
            if let Some(linked) = linked_of(reach) {
                inputs.extend(self.exchange_inputs(linked, &zeros, &zeros, &zeros)?);
            }
            let child = &reaches[reach];
            let bound_forcing = child
                .bind_coupling_tensor(time, end, Some(left), Some(right), &inputs)
                .map_err(channel(reach))?;
            let reservation = child
                .coupling_reservation(&bound_forcing)
                .map_err(channel(reach))?;
            channel_limit_s = channel_limit_s
                .min(reservation.start_cfl_s)
                .min(reservation.face_depletion_limit_s);
            reservations[reach] = Some(reservation);
        }
        let slots = if self
            .linked
            .iter()
            .any(|entry| self.is_ported(entry.reach) || self.is_pool_ended(entry.reach))
        {
            None
        } else {
            Some(self.slot_operands(|reach| {
                reservations[reach].as_ref().ok_or_else(|| {
                    SurfaceChannelError::Numerical("a reach lacks its reservation".into())
                })
            })?)
        };
        let pool_batch = pool_child.map(|child| &child.batch);
        let end_support = self.start_end_support(reaches, pool_batch, reach_ids)?;
        let (stage, wet, initial, law) = if self.link_count == 0 && end_support.is_none() {
            (zeros.clone(), zeros.clone(), zeros, None)
        } else {
            let linked: Vec<&ChannelReach<R>> =
                self.linked.iter().map(|r| &reaches[r.reach]).collect();
            let (stage, wet) = self.node_stages(surface, &linked, pool_batch, reach_ids)?;
            self.check_pumps(&operations.laws, &stage, &wet)?;
            let (regime, law) = if self.link_count == 0 {
                (zeros.clone(), None)
            } else {
                let law = self.raw_weir(operations, (&stage, &wet), LawAt::Start)?;
                let level = self.level_heads(&stage)?;
                let regime = self.level_weir(law.clone(), &stage, &level)?.regime;
                (regime, Some(law))
            };
            let initial = match &end_support {
                // Pool-end volumes and support follow the link volumes and regimes.
                Some(support) => {
                    let ends = Tensor::<R>::zeros(support.shape(), DType::F64, c.device())?;
                    c.cat(&[&stage, &zeros, &ends, &wet, &regime, support], 0)?
                }
                None => c.cat(&[&stage, &zeros, &wet, &regime], 0)?,
            };
            (stage, wet, initial, law)
        };
        let ports = self.port_start(surface, reaches, reach_ids)?;
        Ok(Start {
            time,
            stage,
            wet,
            initial,
            law,
            slots,
            channel_limit_s,
            reservations,
            forcing,
            ports,
            pools: pool_start,
            end_support,
            operations: Arc::clone(operations),
        })
    }

    /// Rejects a running pump that cannot run from the accepted `stage` and `wet` masks.
    ///
    /// No shorter interval changes the accepted state, so both rejections are final:
    /// - `pump-without-suction`: the donor holds no supported water above its suction cutoff.
    /// - `pump-dry-delivery`: the receiver is dry and no declared free outlet lies at or above
    ///   its attachment bed.
    ///
    /// `laws` holds every link law under the operations of the trial.
    fn check_pumps(
        &self,
        laws: &[LinkLaw],
        stage: &Tensor<R>,
        wet: &Tensor<R>,
    ) -> Result<(), SurfaceChannelError> {
        if laws.iter().all(|law| law.pump().is_none_or(|p| !p.running)) {
            return Ok(());
        }
        let stage = stage.try_to_vec::<f64>()?;
        let wet = wet.try_to_vec::<f64>()?;
        let missing = || SurfaceChannelError::Numerical("pump check lacks a group".into());
        for (link, law) in laws.iter().enumerate() {
            let Some(pump) = law.pump().filter(|p| p.running) else {
                continue;
            };
            let groups = *self.link_groups.get(link).ok_or_else(missing)?;
            let side = |index: usize| -> Result<(f64, bool), SurfaceChannelError> {
                let group = groups[index];
                let h = *stage.get(group).ok_or_else(missing)?;
                let w = *wet.get(group).ok_or_else(missing)?;
                Ok((h, w > 0.0 && h > law.bed[index]))
            };
            let id = &self.link_ids[link];
            let donor = pump.donor;
            let (donor_stage, donor_wet) = side(donor)?;
            if !(donor_wet && donor_stage > law.cut[donor]) {
                return Err(SurfaceChannelError::input(format!(
                    "pump-without-suction: structure={id:?} donor stage={donor_stage} holds no supported water above its suction cutoff={}: stop the pump or raise the donor above the cutoff",
                    law.cut[donor]
                )));
            }
            let receiver = 1 - donor;
            let (_, receiver_wet) = side(receiver)?;
            let bed = law.bed[receiver];
            if !receiver_wet && pump.outlet_m.is_none_or(|outlet| bed > outlet) {
                return Err(SurfaceChannelError::input(format!(
                    "pump-dry-delivery: structure={id:?} receiver is dry at bed={bed} with outlet={:?}: declare a free outlet at or above the receiver bed or wet the receiver",
                    pump.outlet_m
                )));
            }
        }
        Ok(())
    }

    /// Slot operands of every linked reach from the reservation that `reservation` returns.
    pub fn slot_operands<'a>(
        &self,
        reservation: impl Fn(usize) -> Result<&'a ChannelCouplingReservation<R>, SurfaceChannelError>,
    ) -> Result<SlotOperands<R>, SurfaceChannelError>
    where
        R: 'a,
    {
        let c = &self.client;
        let mut donors = Vec::with_capacity(self.linked.len());
        let mut outgoing = Vec::with_capacity(self.linked.len());
        let mut velocities = Vec::with_capacity(self.linked.len());
        for entry in &self.linked {
            let reserved = reservation(entry.reach)?;
            let index = &entry.cell_index;
            donors.push(c.index_select(&reserved.donor_volume_m3, 0, index)?);
            outgoing.push(c.index_select(&reserved.outgoing_rate_m3_per_s, 0, index)?);
            velocities.push(c.index_select(&reserved.velocity_m_per_s, 0, index)?);
        }
        let join = |parts: &[Tensor<R>]| -> Result<Tensor<R>, SurfaceChannelError> {
            if parts.is_empty() {
                return Ok(Tensor::<R>::zeros(&[0], DType::F64, c.device())?);
            }
            Ok(c.cat(&parts.iter().collect::<Vec<_>>(), 0)?)
        };
        let surface_zero = Tensor::<R>::zeros(&[self.surface_count], DType::F64, c.device())?;
        let mut group_velocity = vec![surface_zero];
        group_velocity.extend(velocities);
        if self.pool_count > 0 {
            group_velocity.push(Tensor::<R>::zeros(
                &[self.pool_count],
                DType::F64,
                c.device(),
            )?);
        }
        let link_velocity = c.index_select(&join(&group_velocity)?, 0, &self.link_b)?;
        Ok(SlotOperands {
            donor: join(&donors)?,
            outgoing: join(&outgoing)?,
            channel_velocity: [
                c.mul(&link_velocity, &self.tangent[0])?,
                c.mul(&link_velocity, &self.tangent[1])?,
            ],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::SurfaceChannelError;
    use super::super::structures::scenario::{DT, Tanks, TestResult, pump_law, step, structure};
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    /// Input message of the first trial of a running pump between `tanks`.
    fn pump_rejection(tanks: Tanks, direction: DirectionPolicy) -> TestResult<String> {
        let pump = structure(
            "pump",
            pump_law(0.3, None)?,
            direction,
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            [0.0, 0.0],
        );
        match step(&tanks.build(vec![pump])?, DT, &[]) {
            Err(SurfaceChannelError::Input(message)) => Ok(message),
            Err(other) => Err(format!("expected an input rejection, got {other}").into()),
            Ok(_) => Err("the trial admitted a pump that cannot run".into()),
        }
    }

    #[test]
    fn a_running_pump_rejects_without_suction_or_delivery() -> TestResult {
        let message = pump_rejection(Tanks::at(0.3, 0.6), DirectionPolicy::AtoBOnly)?;
        assert!(
            message.contains("pump-without-suction") && message.contains("\"pump\""),
            "{message}"
        );
        let message = pump_rejection(Tanks::at(0.0, 0.6), DirectionPolicy::BtoAOnly)?;
        assert!(
            message.contains("pump-dry-delivery") && message.contains("\"pump\""),
            "{message}"
        );
        Ok(())
    }
}
