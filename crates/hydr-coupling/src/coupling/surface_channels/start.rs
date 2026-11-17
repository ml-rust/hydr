//! Accepted-start operands shared by every attempt of one common trial.

use super::SurfaceChannelError;
use super::context::{Context, Ops};
use super::forcing::BoundForcing;
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
    /// Accepted donor water at every slot, `[C]`.
    pub donor: Tensor<R>,
    /// Accepted native face outflow rate at every slot, `[C]`.
    pub outgoing: Tensor<R>,
    /// Channel-to-surface inflow velocity per link, `[L]` per axis: start velocity on the tangent.
    pub channel_velocity: [Tensor<R>; 2],
    /// Smallest accepted-start CFL or face depletion limit over every reach, in seconds.
    pub channel_limit_s: f64,
    /// Accepted-start reservation of every reach in setup order.
    ///
    /// Every candidate binds through it, so each reach calculates its start fluxes once per trial.
    pub reservations: Vec<ChannelCouplingReservation<R>>,
    pub forcing: BoundForcing<R>,
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
        let (stage, wet) = surface.coupling_cell_stages(&self.links)?;
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
        let mut donors = Vec::with_capacity(self.linked.len());
        let mut outgoing = Vec::with_capacity(self.linked.len());
        let mut velocities = Vec::with_capacity(self.linked.len());
        let mut reserved: Vec<Option<ChannelCouplingReservation<R>>> =
            (0..reaches.len()).map(|_| None).collect();
        let channel = |reach: usize| {
            move |source| SurfaceChannelError::Channel {
                reach_id: reach_ids[reach].to_string(),
                source,
            }
        };
        let linked_of = |reach: usize| self.linked.iter().position(|r| r.reach == reach);
        for &reach in &self.reach_order {
            let bound = &forcing.reaches[reach];
            let mut inputs = bound.lateral.clone();
            if let Some(linked) = linked_of(reach) {
                inputs.extend(self.exchange_inputs(linked, &zeros, &zeros, &zeros)?);
            }
            let child = &reaches[reach];
            let bound_forcing = child
                .bind_coupling_tensor(time, end, Some(&bound.left), Some(&bound.right), &inputs)
                .map_err(channel(reach))?;
            let reservation = child
                .coupling_reservation(&bound_forcing)
                .map_err(channel(reach))?;
            channel_limit_s = channel_limit_s
                .min(reservation.start_cfl_s)
                .min(reservation.face_depletion_limit_s);
            if let Some(linked) = linked_of(reach) {
                let index = &self.linked[linked].cell_index;
                donors.push(c.index_select(&reservation.donor_volume_m3, 0, index)?);
                outgoing.push(c.index_select(&reservation.outgoing_rate_m3_per_s, 0, index)?);
                velocities.push(c.index_select(&reservation.velocity_m_per_s, 0, index)?);
            }
            reserved[reach] = Some(reservation);
        }
        let reservations = reserved
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                SurfaceChannelError::Numerical("a reach lacks its reservation".into())
            })?;
        let linked: Vec<&ChannelReach<R>> = self.linked.iter().map(|r| &reaches[r.reach]).collect();
        let (stage, wet) = self.stages(surface, &linked, reach_ids)?;
        let surface_zero = Tensor::<R>::zeros(&[self.surface_count], DType::F64, c.device())?;
        let mut group_velocity = vec![&surface_zero];
        group_velocity.extend(velocities.iter());
        let link_velocity = c.index_select(&c.cat(&group_velocity, 0)?, 0, &self.link_b)?;
        let channel_velocity = [
            c.mul(&link_velocity, &self.tangent[0])?,
            c.mul(&link_velocity, &self.tangent[1])?,
        ];
        let regime = self.weir(&stage, &wet)?.regime;
        let initial = c.cat(&[&stage, &zeros, &wet, &regime], 0)?;
        Ok(Start {
            time,
            donor: c.cat(&donors.iter().collect::<Vec<_>>(), 0)?,
            outgoing: c.cat(&outgoing.iter().collect::<Vec<_>>(), 0)?,
            stage,
            wet,
            initial,
            channel_velocity,
            channel_limit_s,
            reservations,
            forcing,
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
            .structures
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
