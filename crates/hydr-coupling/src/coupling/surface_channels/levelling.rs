//! Levelling cap: no link volume lifts its receiver above its donor over one interval.
//!
//! Each attempt first completes every child with zero exchange. Those pre-exchange end stages
//! hold every other interval term. Each side then responds to exchange through a plan area that
//! bounds its stage change from above:
//! - Surface lane: the cell area. Stage is linear in water.
//! - Channel slot: cell length times water-surface width. Width never shrinks as stage rises,
//!   so the width at the lowest stage the bound covers gives the largest stage change.
//!
//! For a link from donor `d` to receiver `r`, `O_d` is the donor's summed outgoing weir volume
//! and `I_r` the receiver's summed incoming weir volume. Heads `H` are pre-exchange end stages.
//! The link keeps the fraction `θ = min(1, max(θ_below, θ_level, 0))` of its weir volume:
//! - `θ_below = (z_c - H_r) A_r(H_r) / I_r`: the receiver stays at or below the crest.
//! - `θ_level = (H_d - B) / (O_d / A_d(B) + I_r / A_r(B))`, with `B = max(H_r, z_c)`: the donor
//!   stays at or above the receiver.
//!
//! A `θ` below one shrinks by `16` ulps. Every link then takes the smallest `θ` among the links
//! that share its donor's outgoing total or its receiver's incoming total. Summed exchange at
//! every cell therefore stays within each link's bound.

use super::SurfaceChannelError;
use super::context::{Context, Ops};
use super::evaluate::{Attempt, Trials};
use super::exchange::{Exchange, split};
use super::response::Response;
use super::start::Start;
use crate::channel::{ChannelGeometry, ChannelReach, ChannelReachDefinition};
use crate::solver::Solver;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps, ScatterReduceOp, ShapeOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Relative shrink of a binding levelling factor, in units of `f64::EPSILON`.
const LEVEL_MARGIN_ULPS: f64 = 16.0;

/// Cross-section of one linked channel cell.
#[derive(Clone, Debug)]
pub(super) enum SlotSection {
    Rectangular {
        width_m: f64,
    },
    Triangular {
        slope_sum: f64,
        bed_m: f64,
    },
    /// Ordered `(station, elevation)` profile points, in metres.
    Surveyed {
        points: Vec<(f64, f64)>,
    },
}

/// Plan geometry of one linked channel cell.
#[derive(Clone, Debug)]
pub(super) struct SlotPlan {
    pub length_m: f64,
    pub section: SlotSection,
}

impl SlotPlan {
    /// Plan geometry of cell `cell` of `definition`.
    pub fn new(
        definition: &ChannelReachDefinition,
        cell: usize,
    ) -> Result<Self, SurfaceChannelError> {
        let missing = || {
            SurfaceChannelError::input(format!("reach cell={cell}: link an existing reach cell"))
        };
        Ok(match &definition.geometry {
            ChannelGeometry::Rectangular { cells } => {
                let cell = cells.get(cell).ok_or_else(missing)?;
                Self {
                    length_m: cell.length_m,
                    section: SlotSection::Rectangular {
                        width_m: cell.width_m,
                    },
                }
            }
            ChannelGeometry::PrismaticTriangular {
                left_bank_slope_h_per_v,
                right_bank_slope_h_per_v,
                bed_elevation_m,
                cells,
            } => Self {
                length_m: cells.get(cell).ok_or_else(missing)?.length_m,
                section: SlotSection::Triangular {
                    slope_sum: left_bank_slope_h_per_v + right_bank_slope_h_per_v,
                    bed_m: *bed_elevation_m,
                },
            },
            ChannelGeometry::PrismaticSurveyed { profile, cells, .. } => Self {
                length_m: cells.get(cell).ok_or_else(missing)?.length_m,
                section: SlotSection::Surveyed {
                    points: profile
                        .points
                        .iter()
                        .map(|p| (p.station_m, p.elevation_m))
                        .collect(),
                },
            },
        })
    }

    /// Water-surface width at `stage`, in metres. A bench at `stage` adds no width.
    pub fn width(&self, stage: f64) -> f64 {
        match &self.section {
            SlotSection::Rectangular { width_m } => *width_m,
            SlotSection::Triangular { slope_sum, bed_m } => slope_sum * (stage - bed_m).max(0.0),
            SlotSection::Surveyed { points } => points
                .windows(2)
                .map(|pair| {
                    let &[(x0, z0), (x1, z1)] = pair else {
                        return 0.0;
                    };
                    let (low, high) = (z0.min(z1), z0.max(z1));
                    let run = (x1 - x0).abs();
                    if high < stage {
                        run
                    } else if low < stage {
                        run * (stage - low) / (high - low)
                    } else {
                        0.0
                    }
                })
                .sum(),
        }
    }

    /// Plan area at `stage`, in m².
    pub fn area(&self, stage: f64) -> f64 {
        self.length_m * self.width(stage)
    }
}

/// Pre-exchange end stages and plan areas of one attempt.
pub(super) struct Levelling<R: Runtime<DType = DType>> {
    /// End stage of every group with zero exchange, `[S + C]`.
    pub stage: Tensor<R>,
    /// Surface lane area of every link, `[L]`.
    surface_area: Tensor<R>,
    /// Channel area at its pre-exchange stage, `[L]`. It bounds a channel receiver below the crest.
    channel_start_area: Tensor<R>,
    /// Channel area at `max(channel stage, crest)`, `[L]`. It bounds a channel receiver above it.
    channel_receiver_area: Tensor<R>,
    /// Channel area at `max(surface stage, crest)`, `[L]`. It bounds a channel donor above it.
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
            let crest = *self.crest_m.get(link).ok_or_else(topology)?;
            let plan = self.slot_plans.get(slot).ok_or_else(topology)?;
            let lane_area = *self.lane_area_m2.get(lane).ok_or_else(topology)?;
            let channel_area = plan.area(channel_stage);
            let receiver_area = plan.area(channel_stage.max(crest));
            let donor_area = plan.area(surface_stage.max(crest));
            let shares = [counts[0][link] as f64, counts[1][link] as f64];
            // The largest volume each direction keeps under the cap and the donor limit, with
            // every shared cell split evenly among its links.
            let forward = ((crest - channel_stage) * channel_area / shares[1])
                .max(
                    (surface_stage - channel_stage.max(crest))
                        / (shares[0] / lane_area + shares[1] / receiver_area),
                )
                .max(0.0)
                .min(read(&available, lane)? / shares[0]);
            let backward = ((crest - surface_stage) * lane_area / shares[0])
                .max(
                    (channel_stage - surface_stage.max(crest))
                        / (shares[1] / donor_area + shares[0] / lane_area),
                )
                .max(0.0)
                .min(read(&available, self.surface_count + slot)? / shares[1]);
            bounds[0].push(forward);
            bounds[1].push(backward);
            areas[0].push(lane_area);
            areas[1].push(channel_area);
            areas[2].push(receiver_area);
            areas[3].push(donor_area);
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

    /// Levelling factor of every link for `weir` volumes, `[L]`. Unbound links hold exactly one.
    pub fn level(
        &self,
        weir: &Tensor<R>,
        levelling: &Levelling<R>,
    ) -> Result<Tensor<R>, SurfaceChannelError> {
        let c = &self.client;
        let zero = self.scalar(0.0)?;
        let one = self.scalar(1.0)?;
        let (positive, negative) = split(c, weir, &zero)?;
        let outgoing = self.aggregate(&c.cat(&[&positive, &negative], 0)?)?.value;
        let incoming = self.aggregate(&c.cat(&[&negative, &positive], 0)?)?.value;
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
        let surface_area = &levelling.surface_area;
        let donor_area = pick(surface_area, &levelling.channel_donor_area)?;
        let start_area = pick(&levelling.channel_start_area, surface_area)?;
        let receiver_area = pick(&levelling.channel_receiver_area, surface_area)?;
        let below = c.div(
            &c.mul(&c.sub(&self.crest, &receiver)?, &start_area)?,
            &taken,
        )?;
        let floor = c.maximum(&receiver, &self.crest)?;
        let response = c.add(
            &c.div(&given, &donor_area)?,
            &c.div(&taken, &receiver_area)?,
        )?;
        let level = c.div(&c.sub(&donor, &floor)?, &response)?;
        let theta = c.minimum(&c.maximum(&below, &level)?, &one)?;
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
    use super::*;

    #[test]
    fn surveyed_width_grows_with_stage() {
        let plan = SlotPlan {
            length_m: 2.0,
            section: SlotSection::Surveyed {
                points: vec![(0.0, 4.0), (1.0, 0.0), (3.0, 0.0), (5.0, 4.0)],
            },
        };
        assert_eq!(plan.width(0.0), 0.0);
        assert_eq!(plan.width(2.0), 3.5);
        assert_eq!(plan.width(4.0), 5.0);
        assert_eq!(plan.area(2.0), 7.0);
    }

    #[test]
    fn prismatic_widths_match_their_sections() {
        let rectangle = SlotPlan {
            length_m: 2.0,
            section: SlotSection::Rectangular { width_m: 1.5 },
        };
        assert_eq!(rectangle.area(-3.0), 3.0);
        let triangle = SlotPlan {
            length_m: 2.0,
            section: SlotSection::Triangular {
                slope_sum: 3.0,
                bed_m: 1.0,
            },
        };
        assert_eq!(triangle.width(0.5), 0.0);
        assert_eq!(triangle.width(2.0), 3.0);
    }
}
