//! Reach ends that meet a pool: reservoir operands, end volumes, cut checks, and wet support.

use super::{LinkError, LinkOps};
use crate::numerics::water_transfers::{AggregationKind, TransferTree, finite, reject, upward_add};
use hydr_channel::internal::ChannelReservoirOperands;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, IndexingOps, MidpointOps, ShapeOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Midpoint pool stages and the reservoir head of every pool end.
pub(crate) struct EndHeads<R: Runtime<DType = DType>> {
    /// Temporal midpoint stage per pool, `[P]`.
    pub midpoint: Tensor<R>,
    /// Midpoint stage of each pool end's pool, `[E]`.
    pub heads: Tensor<R>,
}

/// Midpoint stages between `start` and `estimate`, gathered per pool end through `end_pools`.
pub(crate) fn end_heads<R: Runtime<DType = DType>>(
    c: &R::Client,
    start: &Tensor<R>,
    estimate: &Tensor<R>,
    end_pools: &Tensor<R>,
) -> Result<EndHeads<R>, LinkError>
where
    R::Client: LinkOps<R> + MidpointOps<R>,
{
    let midpoint = c.midpoint(start, estimate)?;
    finite(c, &midpoint, "used midpoint pool stages")?;
    let heads = c.index_select(&midpoint, 0, end_pools)?;
    Ok(EndHeads { midpoint, heads })
}

/// Reservoir head and entrance loss at each end of one reach. `None` marks an end without a pool.
pub(crate) struct ReachEnds<R: Runtime<DType = DType>> {
    heads: [Option<Tensor<R>>; 2],
    losses: [Option<Tensor<R>>; 2],
}

impl<R: Runtime<DType = DType>> ReachEnds<R> {
    /// Reads pool end `attachments[side]` from the per-end `heads` and `losses`.
    pub fn new(
        heads: &Tensor<R>,
        losses: &Tensor<R>,
        attachments: [Option<usize>; 2],
    ) -> Result<Self, LinkError> {
        let gather = |values: &Tensor<R>| -> Result<[Option<Tensor<R>>; 2], LinkError> {
            Ok([
                attachments[0].map(|i| values.narrow(0, i, 1)).transpose()?,
                attachments[1].map(|i| values.narrow(0, i, 1)).transpose()?,
            ])
        };
        Ok(Self {
            heads: gather(heads)?,
            losses: gather(losses)?,
        })
    }

    /// Reservoir operands of end `side`, left `0` or right `1`.
    pub fn operands(&self, side: usize) -> Option<ChannelReservoirOperands<'_, R>> {
        self.heads[side]
            .as_ref()
            .zip(self.losses[side].as_ref())
            .map(
                |(elevation_m, entrance_loss_coefficient)| ChannelReservoirOperands {
                    elevation_m,
                    entrance_loss_coefficient,
                },
            )
    }
}

/// Reach end volumes split into pool ends and unconnected ends.
pub(crate) struct EndVolumes<R: Runtime<DType = DType>> {
    /// Outward volume at each pool end, `[E]`.
    pub signed: Tensor<R>,
    /// Outward volume at each unconnected end.
    pub external: Tensor<R>,
    /// Momentum impulse at each pool end, `[E]`.
    pub momentum: Tensor<R>,
}

/// Selects pool-end and unconnected-end values from per-reach `[left, right]` endpoint tensors.
pub(crate) fn end_volumes<R: Runtime<DType = DType>>(
    c: &R::Client,
    outward: &[Tensor<R>],
    momentum: &[Tensor<R>],
    pool_ends: &Tensor<R>,
    external_ends: &Tensor<R>,
) -> Result<EndVolumes<R>, LinkError>
where
    R::Client: LinkOps<R>,
{
    let volumes = c.cat(&outward.iter().collect::<Vec<_>>(), 0)?;
    let momenta = c.cat(&momentum.iter().collect::<Vec<_>>(), 0)?;
    Ok(EndVolumes {
        signed: c.index_select(&volumes, 0, pool_ends)?,
        external: c.index_select(&volumes, 0, external_ends)?,
        momentum: c.index_select(&momenta, 0, pool_ends)?,
    })
}

/// Upper bound of pool exports through attachments at or above each cut.
///
/// `blocks` sums exports per cut block. `covers` sums blocks per cut, with their bounds.
pub(crate) fn cut_demand<R: Runtime<DType = DType>>(
    c: &R::Client,
    blocks: &TransferTree<R>,
    covers: &TransferTree<R>,
    signed: &Tensor<R>,
) -> Result<Tensor<R>, LinkError>
where
    R::Client: LinkOps<R>,
{
    let zero = Tensor::<R>::zeros(&[1], DType::F64, c.device())?;
    let outgoing = c.maximum(&c.neg(signed)?, &zero)?;
    let blocks = blocks.aggregate(c, &outgoing, AggregationKind::Momentum)?;
    let sums = covers.aggregate(c, &blocks.value, AggregationKind::Momentum)?;
    let errors = covers.aggregate(c, &blocks.bound, AggregationKind::Momentum)?;
    let bound = upward_add(
        c,
        &sums.bound,
        &upward_add(c, &errors.value, &errors.bound)?,
    )?;
    Ok(upward_add(c, &sums.value, &bound)?)
}

/// Rejects cut `demand` above the original water `available` above each cut.
pub(crate) fn check_cuts<R: Runtime<DType = DType>>(
    c: &R::Client,
    available: &Tensor<R>,
    demand: &Tensor<R>,
) -> Result<(), LinkError>
where
    R::Client: LinkOps<R>,
{
    finite(c, available, "original cut availability")?;
    finite(c, demand, "cut export upper bound")?;
    reject(
        c,
        &c.gt(demand, available)?,
        "connection exports exceed original water above an attachment cut. Shorten the complete interval",
    )?;
    Ok(())
}

/// One where a pool end's pool holds more water than its attachment cut's upper volume.
pub(crate) fn attachment_wet<R: Runtime<DType = DType>>(
    c: &R::Client,
    pool_volumes: &Tensor<R>,
    cut_upper_volumes: &Tensor<R>,
    end_pools: &Tensor<R>,
    end_cuts: &Tensor<R>,
) -> Result<Tensor<R>, LinkError>
where
    R::Client: LinkOps<R>,
{
    let volume = c.index_select(pool_volumes, 0, end_pools)?;
    let cut_upper = c.index_select(cut_upper_volumes, 0, end_cuts)?;
    // Nominal interpolation can round a below-bed stage above the attachment.
    Ok(c.gt(&volume, &cut_upper)?)
}

/// Rejects an active pool end without wet support at both interval ends.
pub(crate) fn check_support<R: Runtime<DType = DType>>(
    c: &R::Client,
    start_wet: &Tensor<R>,
    end_wet: &Tensor<R>,
    signed: &Tensor<R>,
) -> Result<(), LinkError>
where
    R::Client: LinkOps<R>,
{
    let zero = Tensor::<R>::zeros(&[1], DType::F64, c.device())?;
    let one = Tensor::<R>::full_scalar(&[1], DType::F64, 1.0, c.device())?;
    let active = c.gt(&c.abs(signed)?, &zero)?;
    let supported = c.mul(start_wet, end_wet)?;
    reject(
        c,
        &c.mul(&active, &c.sub(&one, &supported)?)?,
        "active connection loses wet support during the common interval. Shorten the complete interval",
    )?;
    Ok(())
}
