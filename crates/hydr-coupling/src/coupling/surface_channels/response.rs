//! Plan-area response step: the next fixed-point iterate solves each link against its heads.
//!
//! A weir volume changes the end stages it sets. Near level heads the Villemonte factor is
//! steep, so a plain update `x <- g(x)` overshoots and stops contracting. Each linked end stage
//! responds to exchange through the plan areas of the levelling cap: `-1/A` at the donor and
//! `+1/A` at the receiver, per cubic metre. Within one update each link takes the volume `V`
//! at which the weir agrees with that response:
//!
//! `V = K(W(mid(H_start, H + r (V - V_g))))`, with `H` and `V_g` the evaluated end stages and
//! accepted volume.
//!
//! - `K` clamps each direction to the most the levelling cap and the donor limit keep.
//! - `K(W)` falls as `V` rises, so the root lies between `V_g` and the plain proposal
//!   `K(W(mid(H_start, H)))`. Bisection finds it.
//! - Each side's response and bound split a shared cell evenly among its links. Links that
//!   share a cell then never overshoot level together.
//!
//! The next iterate holds `V`, and end stages moved by `r (V - V_g)` summed at every cell.
//! The children then correct the model. Convergence still tests the plain map residual
//! `g(x) - x`, so the accepted fixed point and every budget stay those of the plain update.

use super::SurfaceChannelError;
use super::context::{Context, Ops, transfer_error};
use super::evaluate::{Attempt, Evaluation};
use super::levelling::Levelling;
use super::start::Start;
use crate::numerics::water_transfers::AggregationKind;
use hydr_structures::internal::TensorStructureResult;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Bisection rounds that bound one response step. Each round halves every open bracket.
const ROUND_LIMIT: usize = 64;

/// Stage response of every link's two sides to exchange, each `[L]`.
pub(super) struct Response<R: Runtime<DType = DType>> {
    /// Stage change per cubic metre at the surface lane and the channel slot.
    inverse: [Tensor<R>; 2],
    /// `inverse` times the link count of each side's cell.
    shared: [Tensor<R>; 2],
    /// One where both sides hold a positive plan area.
    modelled: Tensor<R>,
    /// Largest volume the cap and donor limit keep surface-to-channel and channel-to-surface,
    /// each nonnegative.
    bounds: [Tensor<R>; 2],
}

impl<R: Runtime<DType = DType>> Response<R> {
    /// Response of surface and channel plan `areas` with side link `counts`, and the
    /// directional volume `bounds`.
    ///
    /// A side without plan area leaves its link to the plain update.
    pub fn new(
        areas: [&[f64]; 2],
        counts: [&[usize]; 2],
        bounds: [&[f64]; 2],
        device: &R::Device,
    ) -> Result<Self, SurfaceChannelError> {
        let [surface_m2, channel_m2] = areas;
        let inverse = |area: f64| if area > 0.0 { 1.0 / area } else { 0.0 };
        let tensor = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], device);
        let [surface_inverse, channel_inverse] =
            areas.map(|areas| areas.iter().map(|&a| inverse(a)).collect::<Vec<_>>());
        let shared = |inverse: &[f64], count: &[usize]| -> Vec<f64> {
            inverse
                .iter()
                .zip(count)
                .map(|(v, &k)| v * k as f64)
                .collect()
        };
        let modelled: Vec<f64> = surface_m2
            .iter()
            .zip(channel_m2)
            .map(|(&s, &c)| f64::from(u8::from(s > 0.0 && c > 0.0)))
            .collect();
        Ok(Self {
            shared: [
                tensor(&shared(&surface_inverse, counts[0]))?,
                tensor(&shared(&channel_inverse, counts[1]))?,
            ],
            inverse: [tensor(&surface_inverse)?, tensor(&channel_inverse)?],
            modelled: tensor(&modelled)?,
            bounds: [tensor(bounds[0])?, tensor(bounds[1])?],
        })
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Next fixed-point iterate after `evaluation`, by the plan-area response step.
    pub fn respond(
        &self,
        start: &Start<R>,
        attempt: &Attempt<R>,
        levelling: &Levelling<R>,
        evaluation: &Evaluation<R>,
    ) -> Result<Tensor<R>, SurfaceChannelError> {
        // Without links no plan-area response applies: every entry takes the plain update.
        if self.link_count == 0 {
            return Ok(evaluation.packet.clone());
        }
        let c = &self.client;
        let n = self.group_count();
        let l = self.link_count;
        let m = self.transfer_count();
        let response = &levelling.response;
        let packet = &evaluation.packet;
        let end = packet.narrow(0, 0, n)?;
        let accepted = packet.narrow(0, n, l)?;
        // The reach sets each pool-end volume, so it takes the plain update.
        let ends = (m > l)
            .then(|| packet.narrow(0, n + l, m - l))
            .transpose()?;
        let flags = packet.narrow(0, n + m, n + m)?;
        let wet = c.maximum(&start.wet, &packet.narrow(0, n + m, n)?)?;
        let gather = |values: &Tensor<R>, index: &Tensor<R>| c.index_select(values, 0, index);
        let [wet_a, wet_b] = [gather(&wet, &self.link_a)?, gather(&wet, &self.link_b)?];
        let [start_a, start_b] = [
            gather(&start.stage, &self.link_a)?,
            gather(&start.stage, &self.link_b)?,
        ];
        let [end_a, end_b] = [gather(&end, &self.link_a)?, gather(&end, &self.link_b)?];
        let operations = &attempt.operations;
        let structures = operations.structures()?;
        let [forward, backward] = &response.bounds;
        let backward = c.neg(backward)?;
        // Kept weir volume of every link when its own volume moves from `accepted` to `volume`.
        let weir = |volume: &Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
            let shift = c.sub(volume, &accepted)?;
            let surface = c.sub(&end_a, &c.mul(&shift, &response.shared[0])?)?;
            let channel = c.add(&end_b, &c.mul(&shift, &response.shared[1])?)?;
            let law = self.link_weir(
                operations,
                [
                    &c.midpoint(&start_a, &surface)?,
                    &c.midpoint(&start_b, &channel)?,
                ],
                [&wet_a, &wet_b],
                (&levelling.level, false),
            )?;
            let weir = structures.held_volumes(&law, attempt.dt).map_err(|error| {
                SurfaceChannelError::Numerical(format!("link volumes: {error}"))
            })?;
            Ok(c.minimum(&c.maximum(&weir, &backward)?, forward)?)
        };
        let residual = |volume: &Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
            Ok(c.sub(volume, &weir(volume)?)?)
        };
        let one = self.scalar(1.0)?;
        let modelled = &response.modelled;
        let proposal = weir(&accepted)?;
        let rising = c.le(&accepted, &proposal)?;
        let mut lower = c.where_cond(modelled, &c.minimum(&accepted, &proposal)?, &accepted)?;
        let mut upper = c.where_cond(modelled, &c.maximum(&accepted, &proposal)?, &accepted)?;
        let at_accepted = c.sub(&accepted, &proposal)?;
        let at_proposal = residual(&proposal)?;
        let mut residual_lower = c.where_cond(&rising, &at_accepted, &at_proposal)?;
        let mut residual_upper = c.where_cond(&rising, &at_proposal, &at_accepted)?;
        let resolution = c.mul_scalar(&c.sub(&upper, &lower)?, f64::EPSILON)?;
        let zero = self.scalar(0.0)?;
        for _ in 0..ROUND_LIMIT {
            let middle = c.midpoint(&lower, &upper)?;
            let width = c.sub(&upper, &lower)?;
            let scale = c.maximum(&c.abs(&lower)?, &c.abs(&upper)?)?;
            let closed = c.maximum(
                &c.maximum(
                    &c.le(&width, &c.mul_scalar(&scale, f64::EPSILON)?)?,
                    &c.le(&width, &resolution)?,
                )?,
                &c.maximum(&c.eq(&middle, &lower)?, &c.eq(&middle, &upper)?)?,
            )?;
            let open = c.sub(&one, &closed)?;
            if c.max(&open, &[0], false)?.item::<f64>()? == 0.0 {
                break;
            }
            let value = residual(&middle)?;
            let down = c.mul(&open, &c.ge(&value, &zero)?)?;
            let up = c.mul(&open, &c.le(&value, &zero)?)?;
            upper = c.where_cond(&down, &middle, &upper)?;
            residual_upper = c.where_cond(&down, &value, &residual_upper)?;
            lower = c.where_cond(&up, &middle, &lower)?;
            residual_lower = c.where_cond(&up, &value, &residual_lower)?;
        }
        let nearer = c.le(&c.abs(&residual_lower)?, &c.abs(&residual_upper)?)?;
        let volume = c.where_cond(&nearer, &lower, &upper)?;
        let shift = c.sub(&volume, &accepted)?;
        let moved = c.cat(
            &[
                &c.neg(&c.mul(&shift, &response.inverse[0])?)?,
                &c.mul(&shift, &response.inverse[1])?,
            ],
            0,
        )?;
        let change = self
            .donors
            .aggregate(c, &moved, AggregationKind::Momentum)
            .map_err(transfer_error)?
            .value;
        let stages = c.add(&end, &change)?;
        Ok(match &ends {
            Some(ends) => c.cat(&[&stages, &volume, ends, &flags], 0)?,
            None => c.cat(&[&stages, &volume, &flags], 0)?,
        })
    }

    /// Change one rounding of the midpoint heads can cause in each link volume and end stage.
    ///
    /// `law` holds the link laws at `midpoint`, `[S + C + P]`, and `signed` the accepted volumes,
    /// `[L]`.
    /// A link volume `W` changes with a midpoint head by at most `|W| s` per metre, with `s` the
    /// law's head sensitivity. For a weir `s = 1.5 / h_u + 0.385 / gap`: `h_u` is the upstream
    /// head above the crest and `gap` the head difference of submerged flow. One rounding moves
    /// each midpoint head by `ε |H|`. The volume part is `|W| s ε (|H_s| + |H_c|)`. The stage
    /// part sums each link's volume part over its cell's plan area.
    ///
    /// Near level heads the gap term grows without bound. No representable iterate then meets a
    /// budget below this part, so the residual check adds it to each allowance.
    pub fn resolution(
        &self,
        law: &TensorStructureResult<R>,
        signed: &Tensor<R>,
        midpoint: &Tensor<R>,
        levelling: &Levelling<R>,
    ) -> Result<[Tensor<R>; 2], SurfaceChannelError> {
        let c = &self.client;
        let slope = &law.head_sensitivity;
        let gather = |index: &Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
            Ok(c.abs(&c.index_select(midpoint, 0, index)?)?)
        };
        let heads = c.add(&gather(&self.link_a)?, &gather(&self.link_b)?)?;
        let volume = c.mul_scalar(
            &c.mul(&c.mul(&c.abs(signed)?, slope)?, &heads)?,
            f64::EPSILON,
        )?;
        let inverse = &levelling.response.inverse;
        let stage = self
            .donors
            .aggregate(
                c,
                &c.cat(
                    &[&c.mul(&volume, &inverse[0])?, &c.mul(&volume, &inverse[1])?],
                    0,
                )?,
                AggregationKind::Water,
            )
            .map_err(transfer_error)?
            .value;
        Ok([volume, stage])
    }
}

#[cfg(test)]
mod tests {
    use super::super::SurfaceChannelPort;
    use super::super::ports::fixture::{ALPHA, BETA, TestResult};
    use super::super::ports::scenario::{Batch, Reach, Scenario, receipt, step, west};
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    /// Requested duration of every seiche interval, in s. The step limits shorten it.
    const REQUESTED: f64 = 0.25;
    /// Accepted intervals of each seiche run.
    const INTERVALS: usize = 120;
    /// Smallest fraction of its step-limited duration an accepted interval keeps.
    const KEPT: f64 = 0.5;
    /// Fraction of the smallest child step limit a step-limited attempt uses.
    const LIMIT_SAFETY: f64 = 0.95;
    /// Midpoint head difference the run must reach, in m.
    const NEAR_LEVEL_M: f64 = 1e-6;

    /// Smallest step limit of the surface and every reach over the last interval, in s.
    fn step_limit(batch: &Batch) -> TestResult<f64> {
        let surface = batch.surface().last_step()?.ok_or("a surface receipt")?;
        let mut limit = surface.cfl_limit.min(surface.depletion_limit);
        for id in [ALPHA, BETA] {
            let reach = batch.reach(id).ok_or("the reach exists")?;
            let receipt = reach.last_step()?.ok_or("a reach receipt")?;
            limit = limit
                .min(receipt.accepted_start_cfl_limit_s)
                .min(receipt.depletion_limit_s);
        }
        Ok(limit)
    }

    /// A bank link between a floodplain at 1.6 m and a reach at 1.75 m, behind a 1.2 m crest.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every accepted interval keeps at least `KEPT` of `min(REQUESTED, 0.95 × limit)`.
    ///   `limit` is the smallest surface or reach step limit at the interval start.
    /// - Every interval passes its step mass bound and the lifetime composite balance.
    /// - Each side's net exchange matches the link volume within the paired bound.
    /// - Every port receipt pairs within its bound.
    /// - The midpoint heads come within `NEAR_LEVEL_M` of each other.
    fn seiche_keeps_its_step(ports: Vec<SurfaceChannelPort>) -> TestResult {
        let mut case = Scenario::rest(1.0, 1.6, 0.5, ports);
        case.alpha = Reach::rest(0.5, 1.75);
        let mut batch = case.build()?;
        let mut nearest = f64::INFINITY;
        for index in 0..INTERVALS {
            batch = step(&batch, REQUESTED)?;
            let interval = batch.last_step()?.ok_or("an accepted interval")?;
            let dt = interval.end_s - interval.start_s;
            let limited = REQUESTED.min(LIMIT_SAFETY * step_limit(&batch)?);
            assert!(
                dt >= KEPT * limited,
                "interval {index}: dt={dt}, step-limited={limited}, {interval:?}"
            );
            assert!(interval.mass_residual_m3.abs() <= interval.mass_bound_m3);
            let balance = batch.ledger()?;
            assert!(balance.residual_m3.abs() <= balance.bound_m3, "{balance:?}");
            let link = interval.links.first().ok_or("the bank link")?;
            let bound = interval.paired_bound_m3;
            let surface = interval.surface_exchange_out_m3 - interval.surface_exchange_in_m3;
            let channel = interval.channel_exchange_in_m3 - interval.channel_exchange_out_m3;
            assert!(
                (surface - link.signed_volume_m3).abs() <= bound,
                "{interval:?}"
            );
            assert!(
                (channel - link.signed_volume_m3).abs() <= bound,
                "{interval:?}"
            );
            for transfer in &receipt(&batch)?.transfers {
                assert!(transfer.paired_gap_m3.abs() <= transfer.paired_bound_m3);
            }
            let gap = link.midpoint_surface_stage_m - link.midpoint_channel_stage_m;
            nearest = nearest.min(gap.abs());
        }
        assert!(
            nearest <= NEAR_LEVEL_M,
            "nearest midpoint head gap={nearest}"
        );
        Ok(())
    }

    #[test]
    fn a_bank_seiche_keeps_its_step_near_level_heads() -> TestResult {
        seiche_keeps_its_step(Vec::new())
    }

    #[test]
    fn a_ported_bank_seiche_keeps_its_step_through_reversals() -> TestResult {
        seiche_keeps_its_step(vec![west("p")])
    }

    #[test]
    fn shared_cells_scale_their_response_by_link_count() -> Result<(), SurfaceChannelError> {
        let device = CpuDevice::new();
        let response = Response::<CpuRuntime>::new(
            [&[4.0, 4.0], &[8.0, 0.0]],
            [&[2, 2], &[1, 1]],
            [&[1.0, 0.0], &[0.0, 2.0]],
            &device,
        )?;
        let read = |t: &Tensor<CpuRuntime>| t.try_to_vec::<f64>();
        assert_eq!(read(&response.inverse[0])?, [0.25, 0.25]);
        assert_eq!(read(&response.shared[0])?, [0.5, 0.5]);
        assert_eq!(read(&response.inverse[1])?, [0.125, 0.0]);
        // A dry triangular slot holds no plan area, so its link keeps the plain update.
        assert_eq!(read(&response.modelled)?, [1.0, 0.0]);
        Ok(())
    }
}
