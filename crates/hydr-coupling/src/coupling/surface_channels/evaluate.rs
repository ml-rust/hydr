//! One fixed-point map: midpoint heads, weir volumes, levelling caps, donor limits, and both
//! child completions.
//!
//! Every evaluation starts from the accepted children, so no trial state carries over.
//! A child trial whose exchange operands repeat an earlier trial of the attempt in every bit
//! takes that trial's result. See [`TrialMemo`].
//! Momentum closure: surface-to-channel water enters the reach with the surface donor's
//! post-transport velocity projected on the reach tangent. Channel-to-surface water leaves at
//! the channel's accepted-start velocity, the channel's own withdrawal rule, and enters the
//! surface along the tangent. Several inflows into one surface cell carry their summed momentum.

use super::SurfaceChannelError;
use super::context::{Context, Ops, transfer_error};
use super::exchange::Exchange;
use super::levelling::Levelling;
use super::memo::{TrialKeys, TrialMemo};
use super::ports::PortBinding;
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
    /// Channel-to-surface inflow velocity per link, `[L]` per axis.
    pub channel_velocity: [Tensor<R>; 2],
    /// Port fluxes and ported reach reservations of this attempt.
    pub ports: PortBinding<R>,
}

/// Kept child trials of one attempt.
pub(super) type Trials<R> = TrialMemo<Solver<R>, ChannelReach<R>>;

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
    /// Change one rounding of the midpoint heads can cause: in each link volume, `[L]`, and in
    /// each end stage, `[S + C]`. See [`Context::resolution`].
    pub resolution: [Tensor<R>; 2],
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Builds the operands of the attempt ending at `stage.end()`.
    ///
    /// Channel slots reserve native face outflow over the duration and caller withdrawals.
    /// A ported reach's face outflow includes its port export.
    pub fn attempt(
        &self,
        start: &Start<R>,
        stage: SurfaceCouplingStage<R>,
        ports: PortBinding<R>,
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
            channel_velocity: slots.channel_velocity,
            ports,
        })
    }

    /// Evaluates the map at iterate `x` from the accepted `surface` and `reaches`.
    ///
    /// Weir volumes pass the levelling cap of `levelling`, then the donor limit.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &self,
        surface: &Solver<R>,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
        start: &Start<R>,
        attempt: &Attempt<R>,
        levelling: &Levelling<R>,
        x: &Tensor<R>,
        trials: &Trials<R>,
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
        let law = self.weir(&midpoint, &wet, &levelling.level)?;
        let proposed = self
            .link_structures()?
            .held_volumes(&law, attempt.dt)
            .map_err(|error| SurfaceChannelError::Numerical(format!("link volumes: {error}")))?;
        let level = self.level(&proposed, levelling)?;
        let exchange = self.exchange(proposed, level, &attempt.available)?;
        let (next_surface, linked) = self.complete(
            surface, reaches, reach_ids, start, attempt, &exchange, trials,
        )?;
        let borrowed: Vec<&ChannelReach<R>> = linked.iter().collect();
        let (end_stage, end_wet) = self.stages(&next_surface, &borrowed, reach_ids)?;
        let packet = c.cat(&[&end_stage, &exchange.signed, &end_wet, &law.regime], 0)?;
        let resolution = self.resolution(&law, &exchange.signed, &midpoint, levelling)?;
        Ok(Evaluation {
            surface: next_surface,
            linked,
            exchange,
            regime: law.regime,
            midpoint,
            packet,
            resolution,
        })
    }

    /// Completes the surface and every linked reach with the limited volumes of `exchange`.
    ///
    /// A child whose operands repeat a trial kept in `trials` takes the kept result.
    #[allow(clippy::too_many_arguments)]
    pub fn complete(
        &self,
        surface: &Solver<R>,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
        start: &Start<R>,
        attempt: &Attempt<R>,
        exchange: &Exchange<R>,
        trials: &Trials<R>,
    ) -> Result<(Solver<R>, Vec<ChannelReach<R>>), SurfaceChannelError> {
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let to_channel = c.where_cond(&c.gt(&exchange.signed, &zero)?, &exchange.signed, &zero)?;
        let to_surface = c.where_cond(
            &c.lt(&exchange.signed, &zero)?,
            &c.neg(&exchange.signed)?,
            &zero,
        )?;
        let s = self.surface_count;
        let packed = c.cat(
            &[
                &to_channel,
                &to_surface,
                &exchange.incoming.narrow(0, 0, s)?,
                &exchange.outgoing.narrow(0, 0, s)?,
            ],
            0,
        )?;
        let reach_links: Vec<&[usize]> = self.linked.iter().map(|e| e.links.as_slice()).collect();
        let keys = TrialKeys::read(&packed, self.link_count, &reach_links)?;
        let next_surface = trials.surface(&keys.surface, || {
            self.complete_surface(surface, attempt, exchange, &to_surface)
        })?;
        let mut linked = Vec::with_capacity(self.linked.len());
        for ((index, entry), key) in self.linked.iter().enumerate().zip(&keys.linked) {
            let reach = &reaches[entry.reach];
            let next = trials.reach(index, key, || {
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
        Ok((next_surface, linked))
    }

    /// Completes the surface interval with the lane totals of the limited link volumes.
    fn complete_surface(
        &self,
        surface: &Solver<R>,
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
            let momentum = c.mul(to_surface, &attempt.channel_velocity[axis])?;
            let lane = self
                .lanes
                .aggregate(c, &momentum, AggregationKind::Momentum)
                .map_err(transfer_error)?;
            Ok(c.where_cond(&has, &c.div(&lane.value, &safe)?, &zero)?)
        };
        let source = ExchangeSource::new(
            self.surface_links()?,
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
