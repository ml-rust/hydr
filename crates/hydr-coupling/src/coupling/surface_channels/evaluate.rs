//! One fixed-point map: midpoint heads, weir volumes, donor limits, and both child completions.
//!
//! Every evaluation starts from the accepted children, so no trial state carries over.
//! Momentum closure: surface-to-channel water enters the reach with the surface donor's
//! post-transport velocity projected on the reach tangent. Channel-to-surface water leaves at
//! the channel's accepted-start velocity, the channel's own withdrawal rule, and enters the
//! surface along the tangent. Several inflows into one surface cell carry their summed momentum.

use super::SurfaceChannelError;
use super::context::{Context, Ops, transfer_error};
use super::exchange::Exchange;
use super::start::Start;
use crate::channel::ChannelReach;
use crate::numerics::water_transfers::AggregationKind;
use crate::solver::Solver;
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

/// Operands of one attempted common duration.
pub(super) struct Attempt<R: Runtime<DType = DType>> {
    pub end: f64,
    pub dt: f64,
    pub stage: SurfaceCouplingStage<R>,
    /// Reserved donor water at every group, `[S + C]`.
    pub available: Tensor<R>,
    /// Surface-to-channel inflow velocity along each link's reach, `[L]`.
    pub surface_velocity: Tensor<R>,
}

/// Complete candidate children and exchange of one map evaluation.
pub(super) struct Evaluation<R: Runtime<DType = DType>> {
    pub surface: Solver<R>,
    /// Advanced linked reaches in `Context::linked` order.
    pub linked: Vec<ChannelReach<R>>,
    pub exchange: Exchange<R>,
    pub regime: Tensor<R>,
    /// Midpoint stage at every group, `[S + C]`.
    pub midpoint: Tensor<R>,
    /// Output iterate: end stages, signed volumes, end support, regimes.
    pub packet: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Builds the operands of the attempt ending at `stage.end()`.
    ///
    /// Channel slots reserve native face outflow over the duration and caller withdrawals.
    pub fn attempt(
        &self,
        start: &Start<R>,
        stage: SurfaceCouplingStage<R>,
    ) -> Result<Attempt<R>, SurfaceChannelError> {
        let c = &self.client;
        let dt = stage.dt();
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
        let kept = c.mul(&start.donor, &c.sub(&self.scalar(1.0)?, &margin)?)?;
        let reserved = c.mul(
            &c.add(&c.mul_scalar(&start.outgoing, dt)?, &withdrawal)?,
            &c.add_scalar(&margin, 1.0)?,
        )?;
        let remaining = c.maximum(&c.sub(&kept, &reserved)?, &self.scalar(0.0)?)?;
        let available = c.cat(&[stage.available_volume_m3(), &remaining], 0)?;
        let [ux, uy] = stage.velocity_xy();
        let surface_velocity = c.add(
            &c.mul(&c.index_select(ux, 0, &self.link_a)?, &self.tangent[0])?,
            &c.mul(&c.index_select(uy, 0, &self.link_a)?, &self.tangent[1])?,
        )?;
        Ok(Attempt {
            end: stage.end(),
            dt,
            stage,
            available,
            surface_velocity,
        })
    }

    /// Evaluates the map at iterate `x` from the accepted `surface` and `reaches`.
    pub fn evaluate(
        &self,
        surface: &Solver<R>,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
        start: &Start<R>,
        attempt: &Attempt<R>,
        x: &Tensor<R>,
    ) -> Result<Evaluation<R>, SurfaceChannelError> {
        let c = &self.client;
        let n = self.surface_count + self.slot_count;
        let l = self.link_count;
        if x.shape() != [2 * (n + l)] {
            return Err(SurfaceChannelError::Numerical(format!(
                "iterate shape={:?}, expected [{}]: supply the coordinator packet",
                x.shape(),
                2 * (n + l)
            )));
        }
        let midpoint = c.midpoint(&start.stage, &x.narrow(0, 0, n)?)?;
        let end_wet = x.narrow(0, n + l, n)?;
        let wet = c.maximum(&start.wet, &end_wet)?;
        let law = self.weir(&midpoint, &wet)?;
        let proposed = self
            .structures
            .held_volumes(&law, attempt.dt)
            .map_err(|error| SurfaceChannelError::Numerical(format!("link volumes: {error}")))?;
        let exchange = self.exchange(proposed, &attempt.available)?;
        let zero = self.scalar(0.0)?;
        let to_channel = c.where_cond(&c.gt(&exchange.signed, &zero)?, &exchange.signed, &zero)?;
        let to_surface = c.where_cond(
            &c.lt(&exchange.signed, &zero)?,
            &c.neg(&exchange.signed)?,
            &zero,
        )?;
        let next_surface =
            self.complete_surface(surface, start, attempt, &exchange, &to_surface)?;
        let mut linked = Vec::with_capacity(self.linked.len());
        for (index, entry) in self.linked.iter().enumerate() {
            let reach = &reaches[entry.reach];
            let bound = &start.forcing.reaches[entry.reach];
            let mut inputs = bound.lateral.clone();
            inputs.extend(self.exchange_inputs(
                index,
                &to_channel,
                &to_surface,
                &attempt.surface_velocity,
            )?);
            let channel = |source| SurfaceChannelError::Channel {
                reach_id: reach_ids[entry.reach].to_string(),
                source,
            };
            let forcing = reach
                .bind_coupling_reserved(&start.reservations[entry.reach], attempt.end, &inputs)
                .map_err(channel)?;
            linked.push(advanced(
                reach.trial_coupled(attempt.end, &forcing, None, None),
                start.time,
                attempt.end,
                reach_ids[entry.reach],
            )?);
        }
        let borrowed: Vec<&ChannelReach<R>> = linked.iter().collect();
        let (end_stage, end_wet) = self.stages(&next_surface, &borrowed, reach_ids)?;
        let packet = c.cat(&[&end_stage, &exchange.signed, &end_wet, &law.regime], 0)?;
        Ok(Evaluation {
            surface: next_surface,
            linked,
            exchange,
            regime: law.regime,
            midpoint,
            packet,
        })
    }

    /// Completes the surface interval with the lane totals of the limited link volumes.
    fn complete_surface(
        &self,
        surface: &Solver<R>,
        start: &Start<R>,
        attempt: &Attempt<R>,
        exchange: &Exchange<R>,
        to_surface: &Tensor<R>,
    ) -> Result<Solver<R>, SurfaceChannelError> {
        let c = &self.client;
        let s = self.surface_count;
        let inflow = exchange.incoming.narrow(0, 0, s)?;
        let withdrawal = exchange.outgoing.narrow(0, 0, s)?;
        let zero = self.scalar(0.0)?;
        let has = c.gt(&inflow, &zero)?;
        let safe = c.where_cond(&has, &inflow, &self.scalar(1.0)?)?;
        let velocity = |axis: usize| -> Result<Tensor<R>, SurfaceChannelError> {
            let momentum = c.mul(to_surface, &start.channel_velocity[axis])?;
            let lane = self
                .lanes
                .aggregate(c, &momentum, AggregationKind::Momentum)
                .map_err(transfer_error)?;
            Ok(c.where_cond(&has, &c.div(&lane.value, &safe)?, &zero)?)
        };
        let source = ExchangeSource::new(
            &self.links,
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
