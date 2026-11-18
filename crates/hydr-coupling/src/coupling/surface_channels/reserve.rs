//! Donor operands of one evaluation: reserved group water, channel inflow velocities, and the
//! pool-end reservations under the iterate's reservoir heads.

use super::SurfaceChannelError;
use super::context::{Context, Ops};
use super::evaluate::Attempt;
use super::pool_ends::EndBinding;
use super::pools::PoolAttempt;
use super::start::{SlotOperands, Start};
use crate::channel::ChannelReach;
use hydr_channel::internal::ChannelCouplingReservation;
use hydr_solver::internal::SurfaceCouplingStage;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, ShapeOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Donor operands of one evaluation.
pub(super) struct Reserved<R: Runtime<DType = DType>> {
    /// Reserved donor water at every group, `[S + C + P]`.
    pub available: Tensor<R>,
    /// Channel-to-surface inflow velocity per link, `[L]` per axis.
    pub channel_velocity: [Tensor<R>; 2],
    /// Reservoir heads and pool-ended reach reservations, or `None` without pool ends.
    pub ends: Option<EndBinding<R>>,
}

impl<R: Runtime<DType = DType>> Reserved<R> {
    /// The reservation of `reach` in this evaluation: its pool-end reservation, else its port
    /// reservation, else its start reservation.
    pub fn reservation<'a>(
        &'a self,
        start: &'a Start<R>,
        attempt: &'a Attempt<R>,
        reach: usize,
    ) -> Result<&'a ChannelCouplingReservation<R>, SurfaceChannelError> {
        match self.ends.as_ref().and_then(|ends| ends.reservation(reach)) {
            Some(reservation) => Ok(reservation),
            None => attempt.ports.reservation(start, reach),
        }
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Donor water at every group, `[S + C + P]`: post-transport surface water, slot water
    /// after native face outflow over `dt` and the caller withdrawals and margins of `reserve`,
    /// then pool water.
    pub fn donor_water(
        &self,
        stage: &SurfaceCouplingStage<R>,
        slots: &SlotOperands<R>,
        dt: f64,
        [withdrawal, margin]: &[Tensor<R>; 2],
        pools: Option<&PoolAttempt<R>>,
    ) -> Result<Tensor<R>, SurfaceChannelError> {
        let c = &self.client;
        let kept = c.mul(&slots.donor, &c.sub(&self.scalar(1.0)?, margin)?)?;
        let reserved = c.mul(
            &c.add(&c.mul_scalar(&slots.outgoing, dt)?, withdrawal)?,
            &c.add_scalar(margin, 1.0)?,
        )?;
        let remaining = c.maximum(&c.sub(&kept, &reserved)?, &self.scalar(0.0)?)?;
        Ok(match pools {
            Some(pools) => c.cat(
                &[stage.available_volume_m3(), &remaining, &pools.available],
                0,
            )?,
            None => c.cat(&[stage.available_volume_m3(), &remaining], 0)?,
        })
    }

    /// Donor operands of iterate `x`.
    ///
    /// Pool-ended reaches reserve under the reservoir heads of `x`. A linked pool-ended reach
    /// then sets its slots' donor water and inflow velocities. Every other attempt keeps the
    /// operands of `attempt`.
    pub fn reserve(
        &self,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
        start: &Start<R>,
        attempt: &Attempt<R>,
        x: &Tensor<R>,
    ) -> Result<Reserved<R>, SurfaceChannelError> {
        let ends = self.bind_ends(start, attempt, reaches, reach_ids, x)?;
        if let (Some(available), Some(channel_velocity)) =
            (&attempt.available, &attempt.channel_velocity)
        {
            return Ok(Reserved {
                available: available.clone(),
                channel_velocity: channel_velocity.clone(),
                ends,
            });
        }
        let binding = ends.as_ref().ok_or_else(|| {
            SurfaceChannelError::Numerical("an attempt lacks its slot donors".into())
        })?;
        let slots = self.slot_operands(|reach| match binding.reservation(reach) {
            Some(reservation) => Ok(reservation),
            None => attempt.ports.reservation(start, reach),
        })?;
        let available = self.donor_water(
            &attempt.stage,
            &slots,
            attempt.dt,
            &attempt.slot_reserve,
            attempt.pools.as_ref(),
        )?;
        Ok(Reserved {
            available,
            channel_velocity: slots.channel_velocity,
            ends,
        })
    }
}
