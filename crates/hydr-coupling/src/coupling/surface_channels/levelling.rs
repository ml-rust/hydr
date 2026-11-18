//! Levelling caps: no passive link lifts its receiver above its donor over one interval, and no
//! pump draws its donor below suction.
//!
//! Each attempt first completes every child with zero exchange. Those pre-exchange end stages
//! hold every other interval term. Each side then responds to exchange through a plan area that
//! bounds its stage change from above:
//! - Surface lane: the cell area. Stage is linear in water.
//! - Channel slot: cell length times water-surface width at the lowest stage the bound covers.
//!   See [`super::slot_plan`].
//!
//! For a link from donor `d` to receiver `r`, `O_d` is the donor's summed outgoing law volume
//! and `I_r` the receiver's summed incoming law volume. Heads `H` are pre-exchange end stages,
//! and `z_r` the receiver's cutoff. Each law keeps `θ = min(1, max(terms, 0))` of its volume:
//! - Weir, with `z_r` at the crest, and gate, with `z_r` at the sill: `max(θ_below, θ_level)`.
//!   - `θ_below = (z_r - H_r) A_r(H_r) / I_r`: the receiver stays at or below its cutoff, so
//!     free flow stays free.
//!   - `θ_level = (H_d - B) / (O_d / A_d(B) + I_r / A_r(B))`, with `B = max(H_r, z_r)`: the
//!     donor stays at or above the receiver.
//! - Submerged orifice: `θ_level` only. A stage crossing the crown is a law rejection, never a
//!   cap.
//! - Pump: `θ_suction = (H_d - z_s) A_d(z_s) / O_d`, with `z_s` the suction stop. A pump lifts
//!   water, so no levelling term applies.
//!
//! A `θ` below one shrinks by `16` ulps. Every link then takes the smallest `θ` among the links
//! that share its donor's outgoing total or its receiver's incoming total. Summed exchange at
//! every cell therefore stays within each link's bound.

use super::SurfaceChannelError;
use super::context::{Context, Ops};
use super::evaluate::{Attempt, Trials};
use super::exchange::Exchange;
use super::response::Response;
use super::start::Start;
use crate::channel::ChannelReach;
use crate::coupling::links::LinkError;
use crate::coupling::links::donors::split;
use crate::solver::Solver;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps, ScatterReduceOp, ShapeOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Relative shrink of a binding levelling factor, in units of `f64::EPSILON`.
const LEVEL_MARGIN_ULPS: f64 = 16.0;

/// Pre-exchange end stages and plan areas of one attempt.
pub(super) struct Levelling<R: Runtime<DType = DType>> {
    /// End stage of every group with zero exchange, `[S + C]`.
    pub stage: Tensor<R>,
    /// Surface lane area of every link, `[L]`.
    surface_area: Tensor<R>,
    /// Channel area at its pre-exchange stage, `[L]`. It bounds a channel receiver below its
    /// cutoff.
    channel_start_area: Tensor<R>,
    /// Channel area at `max(channel stage, channel cutoff)`, `[L]`. It bounds a channel receiver
    /// above it.
    channel_receiver_area: Tensor<R>,
    /// Channel area at `max(surface stage, surface cutoff)`, `[L]`. It bounds a channel donor
    /// above the surface receiver.
    channel_donor_area: Tensor<R>,
    /// Stage response at the pre-exchange end stages, for the fixed-point response step.
    pub response: Response<R>,
    /// One where the start heads and the pre-exchange end heads of a link each sit within the
    /// head allowance, `[L]`. Nothing else then moves them apart, so the link carries no water.
    pub level: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Completes every child with zero exchange and reads the plan areas at its end stages.
    pub fn levelling(
        &self,
        surface: &Solver<R>,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
        start: &Start<R>,
        attempt: &Attempt<R>,
        trials: &Trials<R>,
    ) -> Result<Levelling<R>, SurfaceChannelError> {
        let c = &self.client;
        let device = c.device();
        let n = self.surface_count + self.slot_count;
        let links = Tensor::<R>::zeros(&[self.link_count], DType::F64, device)?;
        let groups = Tensor::<R>::zeros(&[n], DType::F64, device)?;
        let ones = Tensor::<R>::ones(&[self.link_count], DType::F64, device)?;
        let zero = Exchange {
            proposed: links.clone(),
            level: ones.clone(),
            signed: links,
            factor: ones,
            incoming: groups.clone(),
            outgoing: groups,
            aggregation_bound: 0.0,
        };
        let (next, linked) =
            self.complete(surface, reaches, reach_ids, start, attempt, &zero, trials)?;
        let borrowed: Vec<&ChannelReach<R>> = linked.iter().collect();
        let (stage, _) = self.stages(&next, &borrowed, reach_ids)?;
        let host = stage.try_to_vec::<f64>()?;
        let available = attempt.available.try_to_vec::<f64>()?;
        let count = |cells: &[usize]| -> Vec<usize> {
            cells
                .iter()
                .map(|cell| cells.iter().filter(|&other| other == cell).count())
                .collect()
        };
        let counts = [count(&self.link_lane), count(&self.link_slot)];
        let mut areas = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut bounds = [Vec::new(), Vec::new()];
        let topology = || SurfaceChannelError::Numerical("levelling topology lacks a link".into());
        for (link, (&lane, &slot)) in self.link_lane.iter().zip(&self.link_slot).enumerate() {
            let read = |values: &[f64], index: usize| {
                values.get(index).copied().ok_or_else(|| {
                    SurfaceChannelError::Numerical("pre-exchange stages lack a group".into())
                })
            };
            let surface_stage = read(&host, lane)?;
            let channel_stage = read(&host, self.surface_count + slot)?;
            let law = self.laws.get(link).ok_or_else(topology)?;
            let [surface_cut, channel_cut] = law.cut;
            let plan = self.slot_plans.get(slot).ok_or_else(topology)?;
            let lane_area = *self.lane_area_m2.get(lane).ok_or_else(topology)?;
            // The largest volume each direction keeps under the cap and the donor limit, with
            // every shared cell split evenly among its links.
            let [forward, backward] = law.bounds(
                [surface_stage, channel_stage],
                |side, stage| {
                    if side == 0 {
                        lane_area
                    } else {
                        plan.area(stage)
                    }
                },
                [counts[0][link] as f64, counts[1][link] as f64],
                [
                    read(&available, lane)?,
                    read(&available, self.surface_count + slot)?,
                ],
            );
            bounds[0].push(forward);
            bounds[1].push(backward);
            areas[0].push(lane_area);
            areas[1].push(plan.area(channel_stage));
            areas[2].push(plan.area(channel_stage.max(channel_cut)));
            areas[3].push(plan.area(surface_stage.max(surface_cut)));
        }
        let response = Response::new(
            [&areas[0], &areas[1]],
            [&counts[0], &counts[1]],
            [&bounds[0], &bounds[1]],
            device,
        )?;
        let [surface_area, start_area, receiver_area, donor_area] =
            areas.map(|values| Tensor::<R>::from_slice(&values, &[values.len()], device));
        let level = c.mul(&self.level_heads(&start.stage)?, &self.level_heads(&stage)?)?;
        Ok(Levelling {
            level,
            stage,
            surface_area: surface_area?,
            channel_start_area: start_area?,
            channel_receiver_area: receiver_area?,
            channel_donor_area: donor_area?,
            response,
        })
    }

    /// Levelling factor of every link for law volumes `weir`, `[L]`. Unbound links hold exactly
    /// one.
    pub fn level(
        &self,
        weir: &Tensor<R>,
        levelling: &Levelling<R>,
    ) -> Result<Tensor<R>, SurfaceChannelError> {
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let one = self.scalar(1.0)?;
        let (positive, negative) = split(c, weir, &zero).map_err(LinkError::surface_channel)?;
        let outgoing = self.aggregate(&c.cat(&[&positive, &negative], 0)?)?;
        let incoming = self.aggregate(&c.cat(&[&negative, &positive], 0)?)?;
        let at = |values: &Tensor<R>, index: &Tensor<R>| c.index_select(values, 0, index);
        let forward = c.gt(weir, &zero)?;
        let backward = c.lt(weir, &zero)?;
        let active = c.maximum(&forward, &backward)?;
        let pick = |surface_donor: &Tensor<R>, channel_donor: &Tensor<R>| {
            c.where_cond(&forward, surface_donor, channel_donor)
        };
        let stage = &levelling.stage;
        let donor = pick(&at(stage, &self.link_a)?, &at(stage, &self.link_b)?)?;
        let receiver = pick(&at(stage, &self.link_b)?, &at(stage, &self.link_a)?)?;
        let given = pick(&at(&outgoing, &self.link_a)?, &at(&outgoing, &self.link_b)?)?;
        let taken = pick(&at(&incoming, &self.link_b)?, &at(&incoming, &self.link_a)?)?;
        let given = c.where_cond(&active, &given, &one)?;
        let taken = c.where_cond(&active, &taken, &one)?;
        let cut = pick(&self.cut[1], &self.cut[0])?;
        let surface_area = &levelling.surface_area;
        let donor_area = pick(surface_area, &levelling.channel_donor_area)?;
        let start_area = pick(&levelling.channel_start_area, surface_area)?;
        let receiver_area = pick(&levelling.channel_receiver_area, surface_area)?;
        let below = c.div(&c.mul(&c.sub(&cut, &receiver)?, &start_area)?, &taken)?;
        let floor = c.maximum(&receiver, &cut)?;
        let response = c.add(
            &c.div(&given, &donor_area)?,
            &c.div(&taken, &receiver_area)?,
        )?;
        let level = c.div(&c.sub(&donor, &floor)?, &response)?;
        let passive = c.where_cond(&self.has_below, &c.maximum(&below, &level)?, &level)?;
        let suction = c.div(
            &c.mul(&c.sub(&donor, &self.suction_stop)?, &self.suction_area)?,
            &given,
        )?;
        let theta = c.where_cond(&self.pump, &suction, &passive)?;
        let theta = c.minimum(&theta, &one)?;
        let theta = c.where_cond(&c.gt(&theta, &zero)?, &theta, &zero)?;
        let shrunk = c.mul_scalar(&theta, 1.0 - LEVEL_MARGIN_ULPS * f64::EPSILON)?;
        let theta = c.where_cond(&c.lt(&theta, &one)?, &shrunk, &theta)?;
        let theta = c.where_cond(&active, &theta, &one)?;
        // Each group keeps the smallest factor among links that give from it or take into it.
        let groups = Tensor::<R>::ones(
            &[self.surface_count + self.slot_count],
            DType::F64,
            c.device(),
        )?;
        let from_surface = c.where_cond(&forward, &theta, &one)?;
        let from_channel = c.where_cond(&backward, &theta, &one)?;
        let least = |index: &Tensor<R>, values: &Tensor<R>| {
            c.scatter_reduce(&groups, 0, index, values, ScatterReduceOp::Min, true)
        };
        let gives = c.minimum(
            &least(&self.link_a, &from_surface)?,
            &least(&self.link_b, &from_channel)?,
        )?;
        let takes = c.minimum(
            &least(&self.link_b, &from_surface)?,
            &least(&self.link_a, &from_channel)?,
        )?;
        let shared = pick(
            &c.minimum(&at(&gives, &self.link_a)?, &at(&takes, &self.link_b)?)?,
            &c.minimum(&at(&gives, &self.link_b)?, &at(&takes, &self.link_a)?)?,
        )?;
        Ok(c.where_cond(&active, &shared, &one)?)
    }
}

#[cfg(test)]
mod tests {
    use super::super::structures::scenario::{
        AREA, DT, EQUAL_HEADS, FREE_GATE, HEAD_ALLOWANCE, MIXED_GATE, SUBMERGED_GATE, Tanks,
        TestResult, allowance, check, gate_law, heads, orifice_law, pump_curve, pump_law, receipt,
        step, structure,
    };
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    /// Interval limit of every levelling run.
    const INTERVALS: usize = 200;

    /// An orifice between tanks at `surface` and `reach` until their heads level.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every interval passes its mass bound, both pairings, and the composite balance.
    /// - Every accepted volume keeps the sign of the initial head difference: no overshoot.
    /// - Within `INTERVALS` the link reaches equal heads and carries `0.0`.
    /// - The final heads differ by at most twice the head allowance.
    /// - The moved volume equals half the initial head difference times `AREA` within 1e-9 m³.
    fn orifice_levels(surface: f64, reach: f64) -> TestResult {
        let sign = (surface - reach).signum();
        let culvert = structure(
            "culvert",
            orifice_law(0.1, 0.3)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::Passive,
            [0.0, 0.0],
        );
        let mut batch = Tanks::at(surface, reach).build(vec![culvert])?;
        let mut moved = 0.0;
        let mut levelled = false;
        for _ in 0..INTERVALS {
            batch = step(&batch, DT, &[])?;
            check(&batch)?;
            let link = receipt(&batch, "culvert")?;
            assert!(link.signed * sign >= 0.0, "{link:?}");
            moved += link.signed;
            if link.regime == EQUAL_HEADS {
                assert_eq!(link.signed, 0.0, "{link:?}");
                levelled = true;
                break;
            }
        }
        assert!(levelled, "the orifice never reached equal heads");
        let end = heads(&batch)?;
        assert!((end[0] - end[1]).abs() <= 2.0 * allowance(end), "{end:?}");
        let expected = (surface - reach) * AREA / 2.0;
        assert!((moved - expected).abs() <= 1e-9, "moved={moved}");
        Ok(())
    }

    #[test]
    fn an_orifice_fills_the_reach_to_equal_heads() -> TestResult {
        orifice_levels(1.0, 0.5)
    }

    #[test]
    fn an_orifice_drains_the_reach_to_equal_heads() -> TestResult {
        orifice_levels(0.5, 1.0)
    }

    /// A gate from a deep surface tank into a shallow reach below its sill.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every interval passes its mass bound, both pairings, and the composite balance.
    /// - Every accepted volume is nonnegative.
    /// - Free, mixed, and submerged aperture regimes each hold over some interval.
    /// - Within `INTERVALS` the link reaches equal heads and carries `0.0`.
    #[test]
    fn a_gate_passes_free_mixed_and_submerged_apertures() -> TestResult {
        let gate = structure(
            "gate",
            gate_law(0.2)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::GateOpening { opening_m: 0.2 },
            [0.0, 0.0],
        );
        let mut batch = Tanks::at(1.0, 0.1).build(vec![gate])?;
        let mut seen = Vec::new();
        for _ in 0..INTERVALS {
            batch = step(&batch, DT, &[])?;
            check(&batch)?;
            let link = receipt(&batch, "gate")?;
            assert!(link.signed >= 0.0, "{link:?}");
            if !seen.contains(&link.regime) {
                seen.push(link.regime);
            }
            if link.regime == EQUAL_HEADS {
                assert_eq!(link.signed, 0.0, "{link:?}");
                break;
            }
        }
        for regime in [FREE_GATE, MIXED_GATE, SUBMERGED_GATE, EQUAL_HEADS] {
            assert!(
                seen.contains(&regime),
                "regime {regime} absent from {seen:?}"
            );
        }
        Ok(())
    }

    /// A running pump drains the surface tank into the reach down to its suction bound.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every trial accepts: the suction cap stops the pump without a rejection.
    /// - Every interval passes its mass bound, both pairings, and the composite balance.
    /// - The surface stage never falls below the suction bound `z_s`.
    /// - The suction cap binds in some interval.
    /// - After 40 intervals the surface stage lies in `[z_s, z_s + head_absolute_m]`.
    #[test]
    fn the_suction_cap_stops_a_draining_pump() -> TestResult {
        let suction = 0.3;
        let pump = structure(
            "pump",
            pump_law(suction, None)?,
            DirectionPolicy::AtoBOnly,
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            [0.0, 0.0],
        );
        let mut batch = Tanks::at(0.5, 0.5).build(vec![pump])?;
        let mut capped = false;
        for _ in 0..40 {
            batch = step(&batch, DT, &[])?;
            check(&batch)?;
            let link = receipt(&batch, "pump")?;
            assert!(link.signed >= 0.0, "{link:?}");
            capped |= link.level < 1.0;
            let [surface, _] = heads(&batch)?;
            assert!(surface >= suction, "surface stage={surface}");
        }
        assert!(capped, "the suction cap never bound");
        let [surface, _] = heads(&batch)?;
        assert!(
            surface <= suction + HEAD_ALLOWANCE,
            "surface stage={surface}"
        );
        Ok(())
    }

    /// A pump lifts reach water through a free outlet into a dry surface cell.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The first interval moves `-Q(outlet - midpoint reach stage) dt` within 1e-12 relative.
    /// - The surface cell is wet after it.
    /// - Every one of 10 intervals passes its mass bound, both pairings, and the composite
    ///   balance.
    #[test]
    fn a_pump_delivers_through_its_outlet_into_a_dry_cell() -> TestResult {
        let outlet = 1.5;
        let pump = structure(
            "lift",
            pump_law(0.2, Some(outlet))?,
            DirectionPolicy::BtoAOnly,
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            [1.0, 0.0],
        );
        let tanks = Tanks {
            surface_bed: 1.0,
            surface_stage: 1.0,
            reach_bed: 0.0,
            reach_stage: 0.8,
        };
        let mut batch = tanks.build(vec![pump])?;
        for index in 0..10 {
            batch = step(&batch, DT, &[])?;
            check(&batch)?;
            if index == 0 {
                let interval = batch.last_step()?.ok_or("an accepted interval")?;
                let dt = interval.end_s - interval.start_s;
                let link = receipt(&batch, "lift")?;
                let expected = -pump_curve(outlet - link.midpoint[1]) * dt;
                assert!(
                    (link.signed - expected).abs() <= 1e-12 * expected.abs(),
                    "{link:?}, expected={expected}"
                );
                let [surface, _] = heads(&batch)?;
                assert!(surface > 1.0, "surface stage={surface}");
            }
        }
        Ok(())
    }
}
