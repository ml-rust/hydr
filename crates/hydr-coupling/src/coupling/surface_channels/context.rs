//! Static coordinator topology: link laws, donor trees, surface lanes, channel slots, and pools.

mod access;
mod culverts;
mod nodes;

pub(super) use access::transfer_error;

use super::bindings::channel_cells;
use super::pool_ends::PoolEndTopology;
use super::pools::{LinkSides, PoolParts, PoolTopology};
use super::ports::PortTopology;
use super::runoff::{RunoffTopology, runoff_pool_bindings};
use super::slot_plan::SlotPlan;
use super::structures::{LinkLaw, LinkOperations};
use super::table::{TableEnd, table};
use super::{SurfaceChannelDefinition, SurfaceChannelError};
use crate::channel::ChannelLateralDirection;
use crate::numerics::water_transfers::TransferTree;
use crate::solver::Solver;
use crate::structures::{CulvertClient, StructureElement};
use hydr_solver::internal::SurfaceLinks;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::BTreeMap;
use std::sync::Arc;

pub(super) use super::ops::Ops;
pub(super) use culverts::CulvertStops;
pub(super) use nodes::{LinkedReach, NodeEnd};

/// Immutable coordinator operands built once from a checked definition.
///
/// Donor groups hold the surface lanes, then the channel slots, then the pools.
/// Slots are unique linked `(reach identity, cell)` pairs in ascending order.
pub(super) struct Context<R: Runtime<DType = DType>> {
    pub client: R::Client,
    /// Linked surface cells, or `None` for a coordinator without surface lanes.
    pub links: Option<SurfaceLinks<R>>,
    /// Laws of every bank and structure link under its declared operation.
    ///
    /// A coordinator with operating children replaces them on every attempt.
    pub operations: Arc<LinkOperations<R>>,
    /// Law, direction, and datum of every link.
    pub link_elements: Vec<StructureElement>,
    /// Groups `S + C + P`. Input `e` is link `e` at endpoint A, input `L + e` at endpoint B.
    pub donors: TransferTree<R>,
    /// Groups `S`. Input `e` is link `e` at its surface lane when endpoint A is a surface lane.
    pub lanes: TransferTree<R>,
    pub surface_count: usize,
    pub slot_count: usize,
    pub pool_count: usize,
    pub link_count: usize,
    /// Linked reaches in ascending identity order.
    pub linked: Vec<LinkedReach<R>>,
    /// Setup reach indices without links and without pool ends, in ascending identity order.
    pub unlinked: Vec<usize>,
    /// Every setup reach index in ascending identity order.
    pub reach_order: Vec<usize>,
    /// Donor group of each link's endpoint A, I64 `[L]`.
    pub link_a: Tensor<R>,
    /// Donor group of each link's endpoint B, I64 `[L]`.
    pub link_b: Tensor<R>,
    /// Resolved endpoints `[A, B]` of every link.
    pub link_ends: Vec<[NodeEnd; 2]>,
    /// Donor groups `[A, B]` of every link.
    pub link_groups: Vec<[usize; 2]>,
    /// Surface lane of each link's endpoint A, or `S` without one, I64 `[L]`.
    pub lane_index: Tensor<R>,
    /// Attachment bed of each link's endpoint A, `[L]`. A bank link uses the cell bed.
    pub bed_a: Tensor<R>,
    /// Attachment bed of each link's endpoint B, `[L]`. A bank link uses the cell bed.
    pub bed_b: Tensor<R>,
    /// Cutoff stage of each link's endpoint A and endpoint B, each `[L]`.
    pub cut: [Tensor<R>; 2],
    /// One where a link keeps free flow while its receiver stays at or below its cutoff, `[L]`.
    pub has_below: Tensor<R>,
    /// One at every pump link, `[L]`.
    pub pump: Tensor<R>,
    /// Stage at which the suction cap stops each pump donor, `[L]`. Zero at passive links.
    pub suction_stop: Tensor<R>,
    /// Plan area of each pump donor at its suction stop, `[L]`. Zero at passive links.
    pub suction_area: Tensor<R>,
    /// Invert stops of the culvert links, or `None` without a culvert.
    pub culverts: Option<CulvertStops<R>>,
    /// Unit reach tangent of every bank link, `[L]` per axis. Zero at structure links, whose
    /// water enters either side without velocity.
    pub tangent: [Tensor<R>; 2],
    /// Links with a channel endpoint, ascending, as an I64 index tensor `[Lc]`.
    pub channel_links: Tensor<R>,
    /// One where a link's channel endpoint is B, `[L]`. Zero elsewhere.
    pub channel_b: Tensor<R>,
    /// Net outward sign of each link at the surface, `[L]`: `+1` from A, `-1` from B, else `0`.
    pub surface_sign: Tensor<R>,
    /// Net inward sign of each link at the channel, `[L]`: `+1` into B, `-1` into A, else `0`.
    pub channel_sign: Tensor<R>,
    /// Net inward sign of each link at the pools, `[L]`: `+1` into B, `-1` into A, else `0`.
    pub pool_sign: Tensor<R>,
    /// Position of each channel link's inflow binding in the reach-grouped exchange bindings,
    /// I64 `[Lc]`.
    ///
    /// Linked reaches in identity order contribute their exchange bindings, two per link.
    pub inflow_binding_index: Tensor<R>,
    /// Position of each channel link's withdrawal binding in the reach-grouped exchange bindings.
    pub withdrawal_binding_index: Tensor<R>,
    /// Withdrawal bindings at each slot: exchange links plus caller withdrawals.
    pub slot_withdrawals: Vec<usize>,
    /// Per setup reach: caller withdrawal binding identity to linked slot.
    pub caller_slots: Vec<BTreeMap<String, usize>>,
    /// Link identities in ascending order, bank and structure links together.
    pub link_ids: Vec<String>,
    /// Positions of the bank links in the table, ascending. Public receipts list these only.
    pub bank: Vec<usize>,
    /// Synthesized `(inflow, withdrawal)` channel binding identities per link.
    pub link_bindings: Vec<(String, String)>,
    /// Plan area of every surface lane, in m².
    pub lane_area_m2: Vec<f64>,
    /// Plan geometry of every slot.
    pub slot_plans: Vec<SlotPlan>,
    /// Absolute head allowance, in metres.
    pub head_absolute_m: f64,
    /// Relative head allowance.
    pub head_relative: f64,
    /// Reach-end ports, or `None` without ports.
    pub ports: Option<PortTopology<R>>,
    /// Pool operands, or `None` without pools.
    pub pools: Option<PoolTopology<R>>,
    /// Pool-end operands, or `None` without pool ends.
    pub pool_ends: Option<PoolEndTopology<R>>,
    /// Runoff delivery operands, or `None` without delivery lanes.
    pub runoff: Option<RunoffTopology<R>>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + CulvertClient<R>,
{
    /// Builds every static operand from `definition` for `surface` and the pool child `pools`.
    pub fn new(
        client: R::Client,
        definition: &SurfaceChannelDefinition,
        surface: &Solver<R>,
        pools: Option<&PoolParts<R>>,
    ) -> Result<Self, SurfaceChannelError> {
        let setup = definition.setup();
        let links = if definition.surface_cells().is_empty() {
            None
        } else {
            Some(surface.coupling_links(definition.surface_cells())?)
        };
        let surface_count = definition.surface_cells().len();
        let pool_count = definition.pools.as_ref().map_or(0, |pools| pools.ids.len());
        let mut reach_order: Vec<usize> = (0..setup.reaches.len()).collect();
        reach_order.sort_by(|&a, &b| setup.reaches[a].id.cmp(&setup.reaches[b].id));
        let table = table(definition);
        let mut per_reach: BTreeMap<usize, BTreeMap<usize, Vec<usize>>> = BTreeMap::new();
        for (position, link) in table.iter().enumerate() {
            for end in link.ends {
                if let TableEnd::Channel { reach_index, cell } = end {
                    per_reach
                        .entry(reach_index)
                        .or_default()
                        .entry(cell)
                        .or_default()
                        .push(position);
                }
            }
        }
        let device = client.device().clone();
        let indices = |values: &[usize]| -> Result<Tensor<R>, SurfaceChannelError> {
            let values = values
                .iter()
                .map(|&v| {
                    i64::try_from(v).map_err(|_| {
                        SurfaceChannelError::input("link topology exceeds I64 indexing")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Tensor::<R>::from_slice(&values, &[values.len()], &device)?)
        };
        let mut linked = Vec::new();
        let mut unlinked = Vec::new();
        let mut slot_of = vec![0usize; table.len()];
        let mut slot_beds = Vec::new();
        let mut slot_plans = Vec::new();
        let mut slot_withdrawals = Vec::new();
        let mut caller_slots = vec![BTreeMap::new(); setup.reaches.len()];
        let mut grouped = Vec::with_capacity(table.len());
        for &reach in &reach_order {
            let Some(cells) = per_reach.get(&reach) else {
                // A pool-ended reach completes inside the fixed point.
                if definition
                    .pool_ends
                    .iter()
                    .all(|end| end.reach_index != reach)
                {
                    unlinked.push(reach);
                }
                continue;
            };
            let caller = &setup.reaches[reach].definition;
            let geometry = channel_cells(caller);
            let first_slot = slot_beds.len();
            let mut reach_links = Vec::new();
            for (offset, (&cell, positions)) in cells.iter().enumerate() {
                let (cell_id, bed) = geometry.get(cell).copied().ok_or_else(|| {
                    SurfaceChannelError::input(format!(
                        "reach={:?} cell={cell}: link an existing reach cell",
                        setup.reaches[reach].id
                    ))
                })?;
                let slot = first_slot + offset;
                slot_beds.push(bed);
                slot_plans.push(SlotPlan::new(caller, cell)?);
                let mut withdrawals = positions.len();
                for binding in &caller.lateral_bindings {
                    if binding.cell_id == cell_id
                        && binding.direction == ChannelLateralDirection::Withdrawal
                    {
                        withdrawals += 1;
                        caller_slots[reach].insert(binding.boundary_id.clone(), slot);
                    }
                }
                slot_withdrawals.push(withdrawals);
                for &position in positions {
                    slot_of[position] = slot;
                    reach_links.push(position);
                }
            }
            reach_links.sort_unstable();
            grouped.extend(reach_links.iter().copied());
            let cells: Vec<usize> = cells.keys().copied().collect();
            linked.push(LinkedReach {
                reach,
                cell_index: indices(&cells)?,
                cells,
                links: reach_links,
                caller_bindings: caller.lateral_bindings.len(),
            });
        }
        let slot_count = slot_beds.len();
        let link_count = table.len();
        let floats = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], &device);
        let cells = definition.surface_cells();
        let surface_beds: Vec<f64> = cells
            .iter()
            .map(|&cell| setup.surface.domain.bed()[cell])
            .collect();
        let link_ends: Vec<[NodeEnd; 2]> = table
            .iter()
            .enumerate()
            .map(|(position, link)| {
                link.ends.map(|end| match end {
                    TableEnd::Surface(cell) => {
                        NodeEnd::Surface(cells.partition_point(|&c| c < cell))
                    }
                    TableEnd::Channel { .. } => NodeEnd::Channel(slot_of[position]),
                    TableEnd::Pool(pool) => NodeEnd::Pool(pool),
                })
            })
            .collect();
        let group = |end: NodeEnd| match end {
            NodeEnd::Surface(lane) => lane,
            NodeEnd::Channel(slot) => surface_count + slot,
            NodeEnd::Pool(pool) => surface_count + slot_count + pool,
        };
        let link_groups: Vec<[usize; 2]> = link_ends.iter().map(|ends| ends.map(group)).collect();
        let pool_origin = |pool: usize| {
            definition
                .pools
                .as_ref()
                .and_then(|pools| pools.declared.pools.get(*pools.order.get(pool)?))
                .map_or(0.0, |element| element.curve.origin())
        };
        let beds: Vec<[f64; 2]> = table
            .iter()
            .zip(&link_ends)
            .map(|(link, ends)| {
                link.beds.unwrap_or_else(|| {
                    ends.map(|end| match end {
                        NodeEnd::Surface(lane) => surface_beds[lane],
                        NodeEnd::Channel(slot) => slot_beds[slot],
                        NodeEnd::Pool(pool) => pool_origin(pool),
                    })
                })
            })
            .collect();
        let bed_a: Vec<f64> = beds.iter().map(|b| b[0]).collect();
        let bed_b: Vec<f64> = beds.iter().map(|b| b[1]).collect();
        let mut order = vec![0usize; link_count];
        for (index, &position) in grouped.iter().enumerate() {
            order[position] = index;
        }
        let link_ids: Vec<String> = table.iter().map(|l| l.id.to_string()).collect();
        let mut donor_targets = Vec::with_capacity(2 * link_count);
        let mut lane_targets = Vec::with_capacity(link_count);
        for (e, (id, groups)) in link_ids.iter().zip(&link_groups).enumerate() {
            donor_targets.push((groups[0], e, id.as_str()));
            donor_targets.push((groups[1], link_count + e, id.as_str()));
            if let NodeEnd::Surface(lane) = link_ends[e][0] {
                lane_targets.push((lane, e, id.as_str()));
            }
        }
        let donors = TransferTree::new_grouped(
            &client,
            surface_count + slot_count + pool_count,
            2 * link_count,
            &donor_targets,
        )
        .map_err(transfer_error)?;
        let lanes = TransferTree::new_grouped(&client, surface_count, link_count, &lane_targets)
            .map_err(transfer_error)?;
        let link_elements: Vec<StructureElement> =
            table.iter().map(|link| link.element.clone()).collect();
        let operations = Arc::new(LinkOperations::declared(
            &client,
            &link_elements,
            table.iter().map(|link| link.operation).collect(),
            &beds,
            (setup.config.head_absolute_m, setup.config.head_relative),
        )?);
        let laws = &operations.laws;
        let cell_area = setup.surface.domain.grid().cell_area();
        let lane_area_m2 = vec![cell_area; surface_count];
        let per_law = |value: &dyn Fn(usize, &LinkLaw) -> f64| -> Vec<f64> {
            laws.iter()
                .enumerate()
                .map(|(link, law)| value(link, law))
                .collect()
        };
        let flag = |condition: bool| f64::from(u8::from(condition));
        let pool_tops: Vec<f64> = (0..pool_count)
            .map(|pool| {
                pools
                    .and_then(|parts| parts.plans.get(pool))
                    .map_or(0.0, |plan| plan.top())
            })
            .collect();
        let side_area = |link: usize, side: usize, stage: f64| -> f64 {
            match link_ends[link][side] {
                NodeEnd::Surface(lane) => lane_area_m2.get(lane).copied().unwrap_or(0.0),
                NodeEnd::Channel(slot) => slot_plans.get(slot).map_or(0.0, |plan| plan.area(stage)),
                NodeEnd::Pool(pool) => pools
                    .and_then(|parts| parts.plans.get(pool))
                    .map_or(0.0, |plan| plan.smallest(stage, pool_tops[pool])),
            }
        };
        let cut = |side: usize| per_law(&|_, law| law.cut[side]);
        let stop = per_law(&|_, law| law.pump().map_or(0.0, |p| p.stop_m));
        let stop_area = per_law(&|link, law| {
            law.pump()
                .map_or(0.0, |p| side_area(link, p.donor, p.stop_m))
        });
        let culverts = CulvertStops::new(laws, side_area, &device)?;
        let tangent = |axis: usize| -> Vec<f64> { table.iter().map(|l| l.tangent[axis]).collect() };
        let ports = PortTopology::new(definition, surface)?;
        let channel_links: Vec<usize> = table
            .iter()
            .enumerate()
            .filter_map(|(position, link)| link.channel_side().map(|_| position))
            .collect();
        let sign = |kind: fn(&NodeEnd) -> bool| -> Vec<f64> {
            link_ends
                .iter()
                .map(|[a, b]| match (kind(a), kind(b)) {
                    (true, false) => -1.0,
                    (false, true) => 1.0,
                    _ => 0.0,
                })
                .collect()
        };
        let surface_inward = sign(|end| matches!(end, NodeEnd::Surface(_)));
        let lane_index: Vec<usize> = link_ends
            .iter()
            .map(|ends| match ends[0] {
                NodeEnd::Surface(lane) => lane,
                _ => surface_count,
            })
            .collect();
        let pool_topology = match (pools, &definition.pools) {
            (Some(parts), Some(checked)) => Some(PoolTopology::new(
                &client,
                checked,
                parts,
                (
                    LinkSides {
                        ids: &link_ids,
                        ends: &link_ends,
                        beds: &beds,
                        laws,
                    },
                    &definition.pool_ends,
                ),
                surface_count + slot_count,
                runoff_pool_bindings(definition),
            )?),
            (None, None) => None,
            _ => {
                return Err(SurfaceChannelError::Numerical(
                    "pool operands differ from the declared pools".into(),
                ));
            }
        };
        let pool_ends = PoolEndTopology::new(
            &client,
            &definition.pool_ends,
            (setup.reaches.len(), &reach_order),
            &linked,
            pool_topology.as_ref(),
        )?;
        Ok(Self {
            pool_ends,
            ports,
            slot_plans,
            head_absolute_m: setup.config.head_absolute_m,
            head_relative: setup.config.head_relative,
            link_a: indices(&link_groups.iter().map(|g| g[0]).collect::<Vec<_>>())?,
            link_b: indices(&link_groups.iter().map(|g| g[1]).collect::<Vec<_>>())?,
            lane_index: indices(&lane_index)?,
            bed_a: floats(&bed_a)?,
            bed_b: floats(&bed_b)?,
            cut: [floats(&cut(0))?, floats(&cut(1))?],
            has_below: floats(&per_law(&|_, law| flag(law.has_below())))?,
            pump: floats(&per_law(&|_, law| flag(!law.passive())))?,
            suction_stop: floats(&stop)?,
            suction_area: floats(&stop_area)?,
            culverts,
            bank: table
                .iter()
                .enumerate()
                .filter_map(|(position, link)| link.bank.then_some(position))
                .collect(),
            operations: Arc::clone(&operations),
            link_elements,
            lane_area_m2,
            tangent: [floats(&tangent(0))?, floats(&tangent(1))?],
            channel_b: floats(
                &table
                    .iter()
                    .map(|link| flag(link.channel_side() == Some(1)))
                    .collect::<Vec<_>>(),
            )?,
            surface_sign: floats(&surface_inward.iter().map(|s| -s).collect::<Vec<_>>())?,
            channel_sign: floats(&sign(|end| matches!(end, NodeEnd::Channel(_))))?,
            pool_sign: floats(&sign(|end| matches!(end, NodeEnd::Pool(_))))?,
            inflow_binding_index: indices(
                &channel_links
                    .iter()
                    .map(|&link| 2 * order[link])
                    .collect::<Vec<_>>(),
            )?,
            withdrawal_binding_index: indices(
                &channel_links
                    .iter()
                    .map(|&link| 2 * order[link] + 1)
                    .collect::<Vec<_>>(),
            )?,
            channel_links: indices(&channel_links)?,
            links,
            donors,
            lanes,
            surface_count,
            slot_count,
            pool_count,
            link_count,
            linked,
            unlinked,
            reach_order,
            slot_withdrawals,
            caller_slots,
            link_ids,
            link_ends,
            link_groups,
            link_bindings: table
                .iter()
                .map(|l| (l.bindings.0.to_string(), l.bindings.1.to_string()))
                .collect(),
            pools: pool_topology,
            runoff: RunoffTopology::new(&client, definition, surface)?,
            client,
        })
    }
}
