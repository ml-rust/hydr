//! Levelling caps: no passive link lifts its receiver above its donor over one interval, and no
//! pump draws its donor below suction.
//!
//! Each attempt first completes every child with zero exchange. Those pre-exchange end stages
//! hold every other interval term. Each side then responds to exchange through a plan area that
//! bounds its stage change from above:
//! - Surface lane: the cell area. Stage is linear in water.
//! - Channel slot: cell length times water-surface width at the lowest stage the bound covers.
//!   See [`super::slot_plan`].
//! - Pool: its smallest curve segment area over both link heads, both cutoffs, and a pump's
//!   suction stop. See [`crate::coupling::links::pool_plan`].
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
use super::context::{Context, NodeEnd, Ops};
use super::evaluate::{Accepted, Attempt, Trials};
use super::exchange::Exchange;
use super::response::Response;
use super::slot_plan::SlotPlan;
use super::start::Start;
use super::structures::LinkLaw;
use crate::channel::ChannelReach;
use crate::coupling::links::LinkError;
use crate::coupling::links::donors::split;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps, ScatterReduceOp, ShapeOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Relative shrink of a binding levelling factor, in units of `f64::EPSILON`.
const LEVEL_MARGIN_ULPS: f64 = 16.0;

/// Plan area of one link endpoint as a function of its stage.
#[derive(Clone, Copy)]
enum SideArea<'a> {
    /// One area at every stage: a surface lane, or a pool's smallest covered segment.
    Fixed(f64),
    /// A channel slot's width at the stage.
    Slot(&'a SlotPlan),
}

impl SideArea<'_> {
    fn at(self, stage: f64) -> f64 {
        match self {
            Self::Fixed(area) => area,
            Self::Slot(plan) => plan.area(stage),
        }
    }
}

/// Pre-exchange end stages and plan areas of one attempt.
///
/// Each area pair holds the direction from A to B, then from B to A, each `[L]`.
pub(super) struct Levelling<R: Runtime<DType = DType>> {
    /// End stage of every group with zero exchange, `[S + C + P]`.
    pub stage: Tensor<R>,
    /// Donor area at `B = max(receiver stage, receiver cutoff)`. It bounds a donor above the
    /// receiver.
    donor_area: [Tensor<R>; 2],
    /// Receiver area at its pre-exchange stage. It bounds a receiver below its cutoff.
    below_area: [Tensor<R>; 2],
    /// Receiver area at `B`. It bounds a receiver above its cutoff.
    receiver_area: [Tensor<R>; 2],
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
    ///
    /// Pools take their post-source water plus external inflow at the curve stage, without a
    /// pool trial, so a pool that only links can drain never rejects here.
    pub fn levelling(
        &self,
        accepted: Accepted<'_, R>,
        reach_ids: &[&str],
        start: &Start<R>,
        attempt: &Attempt<R>,
        trials: &Trials<R>,
    ) -> Result<Levelling<R>, SurfaceChannelError> {
        let c = &self.client;
        let device = c.device();
        let n = self.group_count();
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
        let without_pools = Accepted {
            pools: None,
            ..accepted
        };
        let completed = self.complete(without_pools, reach_ids, start, attempt, &zero, trials)?;
        let borrowed: Vec<&ChannelReach<R>> = completed.linked.iter().collect();
        let (stage, _) = self.stages(&completed.surface, &borrowed, reach_ids)?;
        let stage = match &attempt.pools {
            Some(pools) => c.cat(&[&stage, &self.levelling_pool_stages(pools)?], 0)?,
            None => stage,
        };
        let host = stage.try_to_vec::<f64>()?;
        let available = attempt.available.try_to_vec::<f64>()?;
        let mut per_group = vec![0usize; n];
        for groups in &self.link_groups {
            for &group in groups {
                if let Some(count) = per_group.get_mut(group) {
                    *count += 1;
                }
            }
        }
        let counts: [Vec<usize>; 2] = [0, 1].map(|side| {
            self.link_groups
                .iter()
                .map(|groups| per_group.get(groups[side]).copied().unwrap_or(0))
                .collect()
        });
        let mut areas = [Vec::new(), Vec::new()];
        let mut donor = [Vec::new(), Vec::new()];
        let mut below = [Vec::new(), Vec::new()];
        let mut receiver = [Vec::new(), Vec::new()];
        let mut bounds = [Vec::new(), Vec::new()];
        let topology = || SurfaceChannelError::Numerical("levelling topology lacks a link".into());
        let read = |values: &[f64], index: usize| {
            values.get(index).copied().ok_or_else(|| {
                SurfaceChannelError::Numerical("pre-exchange stages lack a group".into())
            })
        };
        for (link, (groups, ends)) in self.link_groups.iter().zip(&self.link_ends).enumerate() {
            let heads = [read(&host, groups[0])?, read(&host, groups[1])?];
            let law = self.laws.get(link).ok_or_else(topology)?;
            let cut = law.cut;
            let side_areas = [0, 1].map(|side| self.side_area(ends[side], heads, law));
            let [area_a, area_b] = side_areas;
            let area_a = area_a.ok_or_else(topology)?;
            let area_b = area_b.ok_or_else(topology)?;
            let area = |side: usize, stage: f64| {
                if side == 0 {
                    area_a.at(stage)
                } else {
                    area_b.at(stage)
                }
            };
            // The largest volume each direction keeps under the cap and the donor limit, with
            // every shared node split evenly among its links.
            let [forward, backward] = law.bounds(
                heads,
                area,
                [counts[0][link] as f64, counts[1][link] as f64],
                [read(&available, groups[0])?, read(&available, groups[1])?],
            );
            bounds[0].push(forward);
            bounds[1].push(backward);
            areas[0].push(area(0, heads[0]));
            areas[1].push(area(1, heads[1]));
            for (direction, (d, r)) in [(0usize, 1usize), (1, 0)].into_iter().enumerate() {
                let floor = heads[r].max(cut[r]);
                donor[direction].push(area(d, floor));
                below[direction].push(area(r, heads[r]));
                receiver[direction].push(area(r, floor));
            }
        }
        let response = Response::new(
            [&areas[0], &areas[1]],
            [&counts[0], &counts[1]],
            [&bounds[0], &bounds[1]],
            device,
        )?;
        let tensors = |values: &[Vec<f64>; 2]| -> Result<[Tensor<R>; 2], SurfaceChannelError> {
            Ok([
                Tensor::<R>::from_slice(&values[0], &[values[0].len()], device)?,
                Tensor::<R>::from_slice(&values[1], &[values[1].len()], device)?,
            ])
        };
        let level = c.mul(&self.level_heads(&start.stage)?, &self.level_heads(&stage)?)?;
        Ok(Levelling {
            level,
            stage,
            donor_area: tensors(&donor)?,
            below_area: tensors(&below)?,
            receiver_area: tensors(&receiver)?,
            response,
        })
    }

    /// Plan area model of one link endpoint at pre-exchange link `heads`.
    ///
    /// A pool takes its smallest segment area over every stage the link's bounds cover: both
    /// heads, both cutoffs, and a pump's suction stop.
    fn side_area(&self, end: NodeEnd, heads: [f64; 2], law: &LinkLaw) -> Option<SideArea<'_>> {
        Some(match end {
            NodeEnd::Surface(lane) => SideArea::Fixed(*self.lane_area_m2.get(lane)?),
            NodeEnd::Channel(slot) => SideArea::Slot(self.slot_plans.get(slot)?),
            NodeEnd::Pool(pool) => {
                let plan = self.pools.as_ref()?.plans.get(pool)?;
                let stop = law.pump().map_or(f64::INFINITY, |p| p.stop_m);
                let low = heads[0]
                    .min(heads[1])
                    .min(law.cut[0])
                    .min(law.cut[1])
                    .min(stop);
                let high = heads[0].max(heads[1]).max(law.cut[0]).max(law.cut[1]);
                SideArea::Fixed(plan.smallest(low, high))
            }
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
        let donor_area = pick(&levelling.donor_area[0], &levelling.donor_area[1])?;
        let start_area = pick(&levelling.below_area[0], &levelling.below_area[1])?;
        let receiver_area = pick(&levelling.receiver_area[0], &levelling.receiver_area[1])?;
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
        let groups = Tensor::<R>::ones(&[self.group_count()], DType::F64, c.device())?;
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
