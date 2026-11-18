//! Static coordinator topology: weir laws, donor trees, surface lanes, and channel slots.

use super::bindings::channel_cells;
use super::levelling::SlotPlan;
use super::ports::PortTopology;
use super::state::LinkCounters;
use super::{SurfaceChannelDefinition, SurfaceChannelError};
use crate::channel::ChannelLateralDirection;
use crate::numerics::water_transfers::{TransferTree, WaterTransferError};
use crate::solver::Solver;
use crate::structures::{
    DirectionPolicy, StructureElement, StructureLaw, StructureOperatingPoint, StructureOperation,
};
use hydr_channel::internal::ChannelCouplingEnd;
use hydr_solver::internal::SurfaceLinks;
use hydr_structures::internal::CouplingStructures;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, HypotOps, IndexingOps, MidpointOps,
    NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};
use std::collections::BTreeMap;

/// Every client operation a common surface/channel interval uses.
pub(super) trait Ops<R: Runtime<DType = DType>>:
    BinaryOps<R>
    + CompareOps<R>
    + ConditionalOps<R>
    + CumulativeOps<R>
    + HypotOps<R>
    + IndexingOps<R>
    + MidpointOps<R>
    + NextRepresentableOps<R>
    + ReduceOps<R>
    + ScalarOps<R>
    + ShapeOps<R>
    + TensorOps<R>
    + TwoSumOps<R>
    + TypeConversionOps<R>
    + UnaryOps<R>
    + BatchedBisectionAlgorithms<R>
    + BatchedNewtonAlgorithms<R>
    + Interp1dAlgorithms<R>
    + AndersonAlgorithms<R>
    + solvr::integrate::CertifiedSimpsonAlgorithms<R>
    + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
    + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
    + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>
{
}

impl<R: Runtime<DType = DType>, T> Ops<R> for T where
    T: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>
{
}

/// A reach that holds at least one link, with its slots and exchange bindings.
pub(super) struct LinkedReach<R: Runtime<DType = DType>> {
    /// Setup reach index.
    pub reach: usize,
    /// Linked cells in ascending order. Their slots are consecutive.
    pub cells: Vec<usize>,
    /// `cells` as an I64 index tensor.
    pub cell_index: Tensor<R>,
    /// Link positions on this reach, in ascending link identity.
    pub links: Vec<usize>,
    /// Caller binding count. Exchange bindings follow, two per link.
    pub caller_bindings: usize,
}

/// Immutable coordinator operands built once from a checked definition.
///
/// Donor groups hold the surface lanes, then the channel slots.
/// Slots are unique linked `(reach identity, cell)` pairs in ascending order.
pub(super) struct Context<R: Runtime<DType = DType>> {
    pub client: R::Client,
    pub links: SurfaceLinks<R>,
    pub structures: CouplingStructures<R>,
    /// Groups `S + C`. Input `e` is link `e` at its surface lane, input `L + e` at its slot.
    pub donors: TransferTree<R>,
    /// Groups `S`. Input `e` is link `e` at its surface lane.
    pub lanes: TransferTree<R>,
    pub surface_count: usize,
    pub slot_count: usize,
    pub link_count: usize,
    /// Linked reaches in ascending identity order.
    pub linked: Vec<LinkedReach<R>>,
    /// Unlinked setup reach indices in ascending identity order.
    pub unlinked: Vec<usize>,
    /// Every setup reach index in ascending identity order.
    pub reach_order: Vec<usize>,
    /// Donor group of each link's surface side, I64 `[L]`.
    pub link_a: Tensor<R>,
    /// Donor group of each link's channel side, I64 `[L]`.
    pub link_b: Tensor<R>,
    /// Bed elevation of every donor group, `[S + C]`.
    pub bed: Tensor<R>,
    pub bed_a: Tensor<R>,
    pub bed_b: Tensor<R>,
    /// Weir crest of every link, `[L]`.
    pub crest: Tensor<R>,
    /// Unit reach tangent of every link, `[L]` per axis.
    pub tangent: [Tensor<R>; 2],
    /// Position of each link's inflow binding in the reach-grouped exchange bindings, I64 `[L]`.
    ///
    /// Linked reaches in identity order contribute their exchange bindings, two per link.
    pub inflow_binding_index: Tensor<R>,
    /// Position of each link's withdrawal binding in the reach-grouped exchange bindings.
    pub withdrawal_binding_index: Tensor<R>,
    /// Withdrawal bindings at each slot: exchange links plus caller withdrawals.
    pub slot_withdrawals: Vec<usize>,
    /// Per setup reach: caller withdrawal binding identity to linked slot.
    pub caller_slots: Vec<BTreeMap<String, usize>>,
    /// Link identities in ascending order.
    pub link_ids: Vec<String>,
    /// Synthesized `(inflow, withdrawal)` channel binding identities per link.
    pub link_bindings: Vec<(String, String)>,
    /// Plan area of every surface lane, in m².
    pub lane_area_m2: Vec<f64>,
    /// Plan geometry of every slot.
    pub slot_plans: Vec<SlotPlan>,
    /// Surface lane of every link.
    pub link_lane: Vec<usize>,
    /// Slot of every link.
    pub link_slot: Vec<usize>,
    /// Weir crest of every link, in metres.
    pub crest_m: Vec<f64>,
    /// Absolute head allowance, in metres.
    pub head_absolute_m: f64,
    /// Relative head allowance.
    pub head_relative: f64,
    /// Reach-end ports, or `None` without ports.
    pub ports: Option<PortTopology<R>>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    /// Builds every static operand from `definition` for `surface`.
    pub fn new(
        client: R::Client,
        definition: &SurfaceChannelDefinition,
        surface: &Solver<R>,
    ) -> Result<Self, SurfaceChannelError> {
        let setup = definition.setup();
        let links = surface.coupling_links(definition.surface_cells())?;
        let surface_count = definition.surface_cells().len();
        let mut reach_order: Vec<usize> = (0..setup.reaches.len()).collect();
        reach_order.sort_by(|&a, &b| setup.reaches[a].id.cmp(&setup.reaches[b].id));
        let mut per_reach: BTreeMap<usize, BTreeMap<usize, Vec<usize>>> = BTreeMap::new();
        for (position, link) in definition.links.iter().enumerate() {
            per_reach
                .entry(link.reach_index)
                .or_default()
                .entry(link.channel_cell)
                .or_default()
                .push(position);
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
        let mut slot_of = vec![0usize; definition.links.len()];
        let mut slot_beds = Vec::new();
        let mut slot_plans = Vec::new();
        let mut slot_withdrawals = Vec::new();
        let mut caller_slots = vec![BTreeMap::new(); setup.reaches.len()];
        let mut grouped = Vec::with_capacity(definition.links.len());
        for &reach in &reach_order {
            let Some(cells) = per_reach.get(&reach) else {
                unlinked.push(reach);
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
        let link_count = definition.links.len();
        let floats = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], &device);
        let surface_beds: Vec<f64> = definition
            .surface_cells()
            .iter()
            .map(|&cell| setup.surface.domain.bed()[cell])
            .collect();
        let lane: Vec<usize> = definition.links.iter().map(|l| l.surface_lane).collect();
        let slot_group: Vec<usize> = slot_of.iter().map(|&s| surface_count + s).collect();
        let bed_a: Vec<f64> = lane.iter().map(|&l| surface_beds[l]).collect();
        let bed_b: Vec<f64> = slot_of.iter().map(|&s| slot_beds[s]).collect();
        let mut order = vec![0usize; link_count];
        for (index, &position) in grouped.iter().enumerate() {
            order[position] = index;
        }
        let link_ids: Vec<String> = definition.links().map(|l| l.id.clone()).collect();
        let mut donor_targets = Vec::with_capacity(2 * link_count);
        let mut lane_targets = Vec::with_capacity(link_count);
        for (e, id) in link_ids.iter().enumerate() {
            donor_targets.push((lane[e], e, id.as_str()));
            donor_targets.push((slot_group[e], link_count + e, id.as_str()));
            lane_targets.push((lane[e], e, id.as_str()));
        }
        let donors = TransferTree::new_grouped(
            &client,
            surface_count + slot_count,
            2 * link_count,
            &donor_targets,
        )
        .map_err(transfer_error)?;
        let lanes = TransferTree::new_grouped(&client, surface_count, link_count, &lane_targets)
            .map_err(transfer_error)?;
        let datum = setup
            .surface
            .domain
            .reference()
            .vertical_datum()
            .to_string();
        let elements = definition
            .links
            .iter()
            .zip(&link_ids)
            .map(|(link, id)| StructureElement {
                id: id.clone(),
                endpoint_a: format!(
                    "surface-cell:{}",
                    setup.links[link.setup_index].surface_cell
                ),
                endpoint_b: format!(
                    "reach:{}:cell:{}",
                    setup.reaches[link.reach_index].id, link.channel_cell
                ),
                datum: datum.clone(),
                direction: DirectionPolicy::Bidirectional,
                law: StructureLaw::SuppressedRectangularWeir(link.weir.clone()),
            })
            .collect();
        let operations = link_ids
            .iter()
            .map(|id| StructureOperation {
                structure_id: id.clone(),
                operating_point: StructureOperatingPoint::Passive,
            })
            .collect();
        let structures = CouplingStructures::new(client.clone(), elements, operations)
            .map_err(|error| SurfaceChannelError::input(format!("link weirs: {error}")))?;
        let crest: Vec<f64> = definition.links.iter().map(|l| l.weir.crest_m()).collect();
        let tangent = |axis: usize| -> Vec<f64> {
            definition.links().map(|l| l.reach_tangent[axis]).collect()
        };
        let mut beds = surface_beds;
        beds.extend_from_slice(&slot_beds);
        let cell_area = setup.surface.domain.grid().cell_area();
        let ports = PortTopology::new(definition, surface)?;
        Ok(Self {
            ports,
            lane_area_m2: vec![cell_area; surface_count],
            slot_plans,
            link_lane: lane.clone(),
            link_slot: slot_of,
            crest_m: crest.clone(),
            head_absolute_m: setup.config.head_absolute_m,
            head_relative: setup.config.head_relative,
            link_a: indices(&lane)?,
            link_b: indices(&slot_group)?,
            bed: floats(&beds)?,
            bed_a: floats(&bed_a)?,
            bed_b: floats(&bed_b)?,
            crest: floats(&crest)?,
            tangent: [floats(&tangent(0))?, floats(&tangent(1))?],
            inflow_binding_index: indices(&order.iter().map(|&g| 2 * g).collect::<Vec<_>>())?,
            withdrawal_binding_index: indices(
                &order.iter().map(|&g| 2 * g + 1).collect::<Vec<_>>(),
            )?,
            links,
            structures,
            donors,
            lanes,
            surface_count,
            slot_count,
            link_count,
            linked,
            unlinked,
            reach_order,
            slot_withdrawals,
            caller_slots,
            link_ids,
            link_bindings: definition
                .links
                .iter()
                .map(|l| (l.inflow_binding.clone(), l.withdrawal_binding.clone()))
                .collect(),
            client,
        })
    }

    /// Zero lifetime link counters.
    pub fn zero_counters(&self) -> Result<LinkCounters<R>, SurfaceChannelError> {
        let zero = Tensor::<R>::zeros(&[self.link_count], DType::F64, self.client.device())?;
        Ok(LinkCounters {
            to_channel_m3: zero.clone(),
            to_surface_m3: zero.clone(),
            roundoff_bound_m3: zero,
        })
    }

    /// Rank-zero F64 `value` on the configured device.
    pub fn scalar(&self, value: f64) -> Result<Tensor<R>, SurfaceChannelError> {
        Ok(Tensor::<R>::full_scalar(
            &[],
            DType::F64,
            value,
            self.client.device(),
        )?)
    }
}

impl<R: Runtime<DType = DType>> Context<R> {
    /// Identity of the port at `end` of setup reach `reach`, if any.
    pub fn port_at(&self, reach: usize, end: ChannelCouplingEnd) -> Option<&str> {
        self.ports.as_ref()?.port_at(reach, end)
    }
}

/// Maps a transfer-tree rejection onto the coordinator taxonomy.
pub(super) fn transfer_error(error: WaterTransferError) -> SurfaceChannelError {
    match error {
        WaterTransferError::Input(message) => SurfaceChannelError::Input(message),
        WaterTransferError::Numerical(message) => SurfaceChannelError::Numerical(message),
        WaterTransferError::Runtime(error) => SurfaceChannelError::Runtime(error),
    }
}
