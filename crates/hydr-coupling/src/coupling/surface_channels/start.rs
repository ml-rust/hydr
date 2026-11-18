//! Accepted-start operands shared by every attempt of one common trial.

use super::SurfaceChannelError;
use super::context::{Context, Ops};
use super::forcing::BoundForcing;
use super::ports::PortStart;
use crate::channel::{ChannelReach, ChannelTensorLateralInput, ChannelTensorWaterAmount};
use crate::solver::Solver;
use hydr_channel::internal::ChannelCouplingReservation;
use hydr_structures::internal::TensorStructureResult;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps, ShapeOps, UnaryOps,
};

/// Structures regime code of submerged weir flow.
const SUBMERGED_CODE: f64 = 3.0;

/// Structures regime code of equal heads above the crest.
const EQUAL_HEADS_CODE: f64 = 5.0;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Operands that depend only on the accepted state and the caller forcing.
pub(super) struct Start<R: Runtime<DType = DType>> {
    pub time: f64,
    /// Accepted stage at every donor group, `[S + C]`.
    pub stage: Tensor<R>,
    /// Accepted wet mask at every donor group, `[S + C]`.
    pub wet: Tensor<R>,
    /// First fixed-point iterate: accepted stages, zero volumes, accepted support and regimes.
    pub initial: Tensor<R>,
    /// Slot operands of every linked reach, or `None` when a linked reach is ported.
    ///
    /// A ported reach's face outflow depends on its port flux, so each attempt builds them.
    pub slots: Option<SlotOperands<R>>,
    /// Smallest accepted-start CFL or face depletion limit over every unported reach, in seconds.
    pub channel_limit_s: f64,
    /// Accepted-start reservation of every unported reach in setup order.
    ///
    /// Every candidate binds through it, so each reach calculates its start fluxes once per trial.
    /// A ported reach reserves once per attempt under its port flux.
    pub reservations: Vec<Option<ChannelCouplingReservation<R>>>,
    pub forcing: BoundForcing<R>,
    /// Ghost states of every port from the accepted reach ends, or `None` without ports.
    pub ports: Option<PortStart<R>>,
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
    pub fn exchange_inputs(
        &self,
        linked: usize,
        to_channel: &Tensor<R>,
        to_surface: &Tensor<R>,
        velocity: &Tensor<R>,
    ) -> Result<Vec<ChannelTensorLateralInput<R>>, SurfaceChannelError> {
        let mut inputs = Vec::new();
        for &link in self.linked.get(linked).map_or(&[][..], |r| &r.links) {
            let (inflow, withdrawal) = &self.link_bindings[link];
            inputs.push(ChannelTensorLateralInput::Inflow {
                boundary_id: inflow.clone(),
                amount: ChannelTensorWaterAmount::IntervalVolume {
                    volume_m3: to_channel.narrow(0, link, 1)?,
                },
                longitudinal_velocity_m_per_s: velocity.narrow(0, link, 1)?,
            });
            inputs.push(ChannelTensorLateralInput::Withdrawal {
                boundary_id: withdrawal.clone(),
                amount: ChannelTensorWaterAmount::IntervalVolume {
                    volume_m3: to_surface.narrow(0, link, 1)?,
                },
            });
        }
        Ok(inputs)
    }

    /// Accepted stages and wet masks: surface lanes, then slots.
    pub fn stages(
        &self,
        surface: &Solver<R>,
        linked: &[&ChannelReach<R>],
        reach_ids: &[&str],
    ) -> Result<(Tensor<R>, Tensor<R>), SurfaceChannelError> {
        let links = self.surface_links()?;
        let (stage, wet) = surface.coupling_cell_stages(links)?;
        let mut stages = vec![stage];
        let mut wets = vec![wet];
        for (entry, reach) in self.linked.iter().zip(linked) {
            let (stage, wet) = reach.coupling_cell_stages(&entry.cells).map_err(|source| {
                SurfaceChannelError::Channel {
                    reach_id: reach_ids[entry.reach].to_string(),
                    source,
                }
            })?;
            stages.push(stage);
            wets.push(wet);
        }
        let c = &self.client;
        Ok((
            c.cat(&stages.iter().collect::<Vec<_>>(), 0)?,
            c.cat(&wets.iter().collect::<Vec<_>>(), 0)?,
        ))
    }

    /// Builds the start operands from the accepted children and the bound forcing.
    pub fn start(
        &self,
        surface: &Solver<R>,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
        end: f64,
        forcing: BoundForcing<R>,
    ) -> Result<Start<R>, SurfaceChannelError> {
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
            if self.is_ported(reach) {
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
        let slots = if self.linked.iter().any(|entry| self.is_ported(entry.reach)) {
            None
        } else {
            Some(self.slot_operands(|reach| {
                reservations[reach].as_ref().ok_or_else(|| {
                    SurfaceChannelError::Numerical("a reach lacks its reservation".into())
                })
            })?)
        };
        let (stage, wet, initial) = if self.link_count == 0 {
            (zeros.clone(), zeros.clone(), zeros)
        } else {
            let linked: Vec<&ChannelReach<R>> =
                self.linked.iter().map(|r| &reaches[r.reach]).collect();
            let (stage, wet) = self.stages(surface, &linked, reach_ids)?;
            let regime = self.weir(&stage, &wet)?.regime;
            let initial = c.cat(&[&stage, &zeros, &wet, &regime], 0)?;
            (stage, wet, initial)
        };
        let ports = self.port_start(surface, reaches, reach_ids)?;
        Ok(Start {
            time,
            stage,
            wet,
            initial,
            slots,
            channel_limit_s,
            reservations,
            forcing,
            ports,
        })
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

    /// Weir law of every link at group `stage` with support `wet`, both `[S + C]`.
    ///
    /// Submerged heads that differ by at most the head allowance count as equal heads and carry
    /// no water. The allowance is `head_absolute_m + head_relative * max(|H_a|, |H_b|)`.
    pub fn weir(
        &self,
        stage: &Tensor<R>,
        wet: &Tensor<R>,
    ) -> Result<TensorStructureResult<R>, SurfaceChannelError> {
        let c = &self.client;
        let supported = c.mul(wet, &c.gt(stage, &self.bed)?)?;
        let gather = |values: &Tensor<R>, index: &Tensor<R>| c.index_select(values, 0, index);
        let surface = gather(stage, &self.link_a)?;
        let channel = gather(stage, &self.link_b)?;
        let mut law = self
            .link_structures()?
            .evaluate_heads(
                &surface,
                &channel,
                &self.bed_a,
                &self.bed_b,
                &gather(&supported, &self.link_a)?,
                &gather(&supported, &self.link_b)?,
            )
            .map_err(|error| SurfaceChannelError::Numerical(format!("link weir: {error}")))?;
        let magnitude = c.maximum(&c.abs(&surface)?, &c.abs(&channel)?)?;
        let allowance = c.add_scalar(
            &c.mul_scalar(&magnitude, self.head_relative)?,
            self.head_absolute_m,
        )?;
        let level = c.mul(
            &c.eq(&law.regime, &self.scalar(SUBMERGED_CODE)?)?,
            &c.le(&c.abs(&c.sub(&surface, &channel)?)?, &allowance)?,
        )?;
        law.regime = c.where_cond(&level, &self.scalar(EQUAL_HEADS_CODE)?, &law.regime)?;
        law.discharge = c.where_cond(&level, &self.scalar(0.0)?, &law.discharge)?;
        Ok(law)
    }
}
