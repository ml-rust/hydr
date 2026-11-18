//! Static pool operands: plans, capacities, inflow assembly, attachment cuts, and cutoffs.

use super::super::SurfaceChannelError;
use super::super::context::{Context, NodeEnd, Ops};
use super::super::pool_ends::CheckedPoolEnd;
use super::super::structures::LinkLaw;
use super::CheckedPools;
use super::child::{PoolFns, PoolParts, inflow_error, pool_error};
use crate::coupling::level_pools::InflowContext;
use crate::coupling::links::LinkError;
use crate::coupling::links::cuts::{CutAttachment, CutCover, cut_cover};
use crate::coupling::links::pool_ends;
use crate::coupling::links::pool_heads::{Heads, PoolAttachments};
use crate::coupling::links::pool_plan::PoolPlan;
use crate::numerics::water_transfers::TransferTree;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Both sides of every link of the table, in link order.
pub(in super::super) struct LinkSides<'a> {
    pub ids: &'a [String],
    pub ends: &'a [[NodeEnd; 2]],
    /// Attachment beds `[A, B]`, in metres.
    pub beds: &'a [[f64; 2]],
    pub laws: &'a [LinkLaw],
}

/// Attachment cuts of every pool and the operands that sum exports above them.
pub(in super::super) struct PoolCuts<R: Runtime<DType = DType>> {
    /// Sums attachment exports per aligned block.
    pub blocks: TransferTree<R>,
    /// Sums blocks per cut.
    pub covers: TransferTree<R>,
    /// Pool of every cut, I64 `[K]`.
    pub pool: Tensor<R>,
    /// Upper bound of the volume below every cut, `[K]`.
    pub upper: Tensor<R>,
    /// Link of every pool attachment, I64 `[Q]`.
    pub link: Tensor<R>,
    /// Sign that turns each link volume into an attachment volume, negative out of the pool, `[Q]`.
    pub sign: Tensor<R>,
}

/// Immutable pool operands of one coordinator.
pub(in super::super) struct PoolTopology<R: Runtime<DType = DType>> {
    /// Pool identities in ascending order.
    pub ids: Vec<String>,
    /// Plan areas of every pool.
    pub plans: Vec<PoolPlan>,
    /// Curve origin of every pool, `[P]`.
    pub origin: Tensor<R>,
    /// Top curve knot volume of every pool, `[P]`.
    pub maximum: Tensor<R>,
    /// Top curve knot volume of every pool, in m³.
    pub maximum_m3: Vec<f64>,
    /// External inflow bindings and their assembly.
    pub inflows: InflowContext<R>,
    /// Pool calls bound to this runtime.
    pub fns: PoolFns<R>,
    /// The pools declare source policies.
    pub sources: bool,
    /// Pool-side cutoff of each link's endpoint A, `-inf` at other nodes, `[L]`.
    pub cutoff_a: Tensor<R>,
    /// Pool-side cutoff of each link's endpoint B, `-inf` at other nodes, `[L]`.
    pub cutoff_b: Tensor<R>,
    /// Curve origin of every pool group, zero at other groups, `[S + C + P]`.
    pub group_origin: Tensor<R>,
    /// Attachment cuts, or `None` without pool attachments.
    pub cuts: Option<PoolCuts<R>>,
    /// Attachment cut of every pool end, in pool-end order.
    pub end_cuts: Vec<usize>,
    empty_index: Tensor<R>,
    empty: Tensor<R>,
    zero: Tensor<R>,
}

impl<R: Runtime<DType = DType>> PoolTopology<R>
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
    /// Operands of `pools` with the child `parts`, the link table, and the pool ends.
    ///
    /// Pool groups start at `offset`.
    pub fn new(
        client: &R::Client,
        pools: &CheckedPools,
        parts: &PoolParts<R>,
        (links, owned_ends): (LinkSides<'_>, &[CheckedPoolEnd]),
        offset: usize,
    ) -> Result<Self, SurfaceChannelError> {
        let LinkSides {
            ids,
            ends,
            beds,
            laws,
        } = links;
        let device = client.device();
        let floats = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], device);
        let index = |values: &[usize]| -> Result<Tensor<R>, SurfaceChannelError> {
            let values = values
                .iter()
                .map(|&v| {
                    i64::try_from(v).map_err(|_| {
                        SurfaceChannelError::input("pool topology exceeds I64 indexing")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Tensor::<R>::from_slice(&values, &[values.len()], device)?)
        };
        let elements = parts.batch.elements();
        let origins: Vec<f64> = elements.iter().map(|e| e.curve.origin()).collect();
        let maximum_m3: Vec<f64> = elements
            .iter()
            .map(|e| e.curve.maximum_volume_m3())
            .collect();
        let group_count = offset + origins.len();
        let mut group_origin = vec![0.0; group_count];
        group_origin[offset..].copy_from_slice(&origins);
        let mut attachments = Vec::new();
        let mut attachment_link = Vec::new();
        let mut attachment_sign = Vec::new();
        let mut cutoffs = [Vec::new(), Vec::new()];
        for (link, (link_ends, law)) in ends.iter().zip(laws).enumerate() {
            for (side, end) in link_ends.iter().enumerate() {
                let NodeEnd::Pool(pool) = *end else {
                    cutoffs[side].push(f64::NEG_INFINITY);
                    continue;
                };
                cutoffs[side].push(law.cut[side]);
                attachments.push(CutAttachment {
                    pool,
                    bed_m: beds[link][side],
                    id: ids[link].as_str(),
                });
                attachment_link.push(link);
                attachment_sign.push(if side == 0 { -1.0 } else { 1.0 });
            }
        }
        // A pool end attaches at its end cell bed. Its volume enters the cut values after every
        // link volume and is positive into the pool.
        let first_end = attachments.len();
        for (position, end) in owned_ends.iter().enumerate() {
            attachments.push(CutAttachment {
                pool: end.pool,
                bed_m: end.bed_m,
                id: end.id.as_str(),
            });
            attachment_link.push(ids.len() + position);
            attachment_sign.push(1.0);
        }
        let mut end_cuts = Vec::new();
        let cuts = if attachments.is_empty() {
            None
        } else {
            let CutCover {
                cuts,
                blocks,
                covers,
                attachment_cuts,
            } = cut_cover(client, origins.len(), &attachments)
                .map_err(LinkError::surface_channel)?;
            end_cuts = attachment_cuts.get(first_end..).unwrap_or(&[]).to_vec();
            let upper = (parts.fns.cut_volumes)(&parts.batch, &cuts).map_err(pool_error)?;
            Some(PoolCuts {
                blocks,
                covers,
                pool: index(&cuts.iter().map(|cut| cut.pool_index).collect::<Vec<_>>())?,
                upper,
                link: index(&attachment_link)?,
                sign: floats(&attachment_sign)?,
            })
        };
        let inflows =
            InflowContext::new(client, elements, &pools.declared.inflows).map_err(inflow_error)?;
        Ok(Self {
            ids: pools.ids.clone(),
            plans: parts.plans.clone(),
            origin: floats(&origins)?,
            maximum: floats(&maximum_m3)?,
            maximum_m3,
            inflows,
            fns: parts.fns,
            sources: pools.declared.sources.is_some(),
            cutoff_a: floats(&cutoffs[0])?,
            cutoff_b: floats(&cutoffs[1])?,
            group_origin: floats(&group_origin)?,
            cuts,
            end_cuts,
            empty_index: index(&[])?,
            empty: floats(&[])?,
            zero: Tensor::<R>::zeros(&[], DType::F64, device)?,
        })
    }
}

impl<R: Runtime<DType = DType>> PoolTopology<R> {
    /// Pool-local attachments without links: wet flags and the midpoint depth check.
    pub fn local(&self) -> PoolAttachments<'_, R> {
        PoolAttachments {
            a: &self.empty_index,
            b: &self.empty_index,
            bed_a: &self.empty,
            bed_b: &self.empty,
            cutoff_a: &self.empty,
            cutoff_b: &self.empty,
            origin: &self.origin,
            zero: &self.zero,
        }
    }
}

impl<R: Runtime<DType = DType>> Context<R> {
    /// Pool operands. A coordinator without pools rejects.
    pub fn pool_topology(&self) -> Result<&PoolTopology<R>, SurfaceChannelError> {
        self.pools.as_ref().ok_or_else(|| {
            SurfaceChannelError::Numerical(
                "a coordinator without pools has no pool operands".into(),
            )
        })
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Rejects a wet pool whose temporal midpoint stage between `start` and `end` sits at or below
    /// its origin. Each argument is `[S + C + P]`.
    pub fn check_pool_midpoints(
        &self,
        start: &Tensor<R>,
        end: &Tensor<R>,
        start_wet: &Tensor<R>,
        end_wet: &Tensor<R>,
    ) -> Result<(), SurfaceChannelError> {
        let Some(topology) = &self.pools else {
            return Ok(());
        };
        let offset = self.surface_count + self.slot_count;
        let pools = |values: &Tensor<R>| values.narrow(0, offset, self.pool_count);
        topology
            .local()
            .midpoint_heads(
                &self.client,
                &pools(start)?,
                &pools(end)?,
                &pools(start_wet)?,
                &pools(end_wet)?,
            )
            .map_err(LinkError::surface_channel)?;
        Ok(())
    }

    /// Rejects an active link whose pool donor ends at or below a cutoff raised above the pool
    /// origin. `end` and `wet` are end stages and wet masks, `[S + C + P]`.
    pub fn check_pool_crossing(
        &self,
        end: &Tensor<R>,
        wet: &Tensor<R>,
        signed: &Tensor<R>,
    ) -> Result<(), SurfaceChannelError> {
        let Some(topology) = &self.pools else {
            return Ok(());
        };
        let attachments = PoolAttachments {
            a: &self.link_a,
            b: &self.link_b,
            bed_a: &self.bed_a,
            bed_b: &self.bed_b,
            cutoff_a: &topology.cutoff_a,
            cutoff_b: &topology.cutoff_b,
            origin: &topology.group_origin,
            zero: &topology.zero,
        };
        let c = &self.client;
        let heads: Heads<R> = attachments
            .heads(c, end, wet)
            .map_err(LinkError::surface_channel)?;
        attachments
            .check_raised_crossing(c, &heads, signed)
            .map_err(LinkError::surface_channel)
    }

    /// Rejects exports through pool attachments at or above a cut beyond the pool water above it.
    ///
    /// `available` holds each pool's post-source water, `[P]`, `signed` the accepted link
    /// volumes, `[L]`, and `ends` the accepted pool-end volumes, `[E]`, positive into the pools.
    /// Incoming water funds no export.
    pub fn check_pool_cuts(
        &self,
        available: &Tensor<R>,
        signed: &Tensor<R>,
        ends: Option<&Tensor<R>>,
    ) -> Result<(), SurfaceChannelError> {
        let Some(cuts) = self.pools.as_ref().and_then(|p| p.cuts.as_ref()) else {
            return Ok(());
        };
        let c = &self.client;
        let joined;
        let volumes = match ends {
            Some(ends) => {
                joined = c.cat(&[signed, ends], 0)?;
                &joined
            }
            None => signed,
        };
        let values = c.mul(&c.index_select(volumes, 0, &cuts.link)?, &cuts.sign)?;
        let demand = pool_ends::cut_demand(c, &cuts.blocks, &cuts.covers, &values)
            .map_err(LinkError::surface_channel)?;
        let above = c.sub(&c.index_select(available, 0, &cuts.pool)?, &cuts.upper)?;
        let rounding = c.add_scalar(
            &c.mul_scalar(&c.abs(&above)?, 2.0 * f64::EPSILON)?,
            f64::from_bits(1),
        )?;
        let lower = c.maximum(&c.sub(&above, &rounding)?, &self.scalar(0.0)?)?;
        pool_ends::check_cuts(c, &lower, &demand).map_err(LinkError::surface_channel)
    }
}
