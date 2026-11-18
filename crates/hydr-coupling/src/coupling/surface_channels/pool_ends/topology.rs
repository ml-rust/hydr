//! Static pool-end operands: owned reach ends, pools, attachment cuts, and the pool sums.

use super::CheckedPoolEnd;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::{LinkedReach, transfer_error};
use crate::coupling::surface_channels::pools::PoolTopology;
use crate::numerics::water_transfers::TransferTree;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Immutable pool-end operands of one coordinator, in ascending pool-end identity.
pub(in super::super) struct PoolEndTopology<R: Runtime<DType = DType>> {
    /// Admitted pool ends.
    pub ends: Vec<CheckedPoolEnd>,
    /// Per setup reach: the pool end at its `[left, right]` end.
    pub reach_ends: Vec<[Option<usize>; 2]>,
    /// Setup reaches with a pool end and no link, in ascending reach identity.
    ///
    /// Each one completes inside the fixed point, after the linked reaches.
    pub ended: Vec<usize>,
    /// Pool of every end, I64 `[E]`.
    pub pools: Tensor<R>,
    /// Attachment cut of every end, I64 `[E]`.
    pub cuts: Tensor<R>,
    /// Entrance loss of every end, `[E]`.
    pub losses: Tensor<R>,
    /// Sums end volumes per pool, `[P]` from `[E]`.
    pub tree: TransferTree<R>,
}

impl<R: Runtime<DType = DType>> PoolEndTopology<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    /// Operands of `ends`, or `None` without pool ends.
    ///
    /// `reaches` is the setup reach count and `order` every setup reach in ascending identity.
    /// `pools` holds the attachment cut of every end.
    pub fn new(
        client: &R::Client,
        ends: &[CheckedPoolEnd],
        (reaches, order): (usize, &[usize]),
        linked: &[LinkedReach<R>],
        pools: Option<&PoolTopology<R>>,
    ) -> Result<Option<Self>, SurfaceChannelError> {
        if ends.is_empty() {
            return Ok(None);
        }
        let pools = pools.ok_or_else(|| {
            SurfaceChannelError::Numerical("pool ends lack their pool operands".into())
        })?;
        if pools.end_cuts.len() != ends.len() {
            return Err(SurfaceChannelError::Numerical(
                "pool ends differ from their attachment cuts".into(),
            ));
        }
        let mut reach_ends = vec![[None; 2]; reaches];
        for (position, end) in ends.iter().enumerate() {
            reach_ends[end.reach_index][end.side] = Some(position);
        }
        let ended = order
            .iter()
            .copied()
            .filter(|&reach| {
                reach_ends[reach].iter().any(Option::is_some)
                    && linked.iter().all(|entry| entry.reach != reach)
            })
            .collect();
        let device = client.device();
        let index = |values: &[usize]| -> Result<Tensor<R>, SurfaceChannelError> {
            let values = values
                .iter()
                .map(|&v| {
                    i64::try_from(v).map_err(|_| {
                        SurfaceChannelError::input("pool-end topology exceeds I64 indexing")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Tensor::<R>::from_slice(&values, &[values.len()], device)?)
        };
        let targets: Vec<(usize, usize, &str)> = ends
            .iter()
            .enumerate()
            .map(|(position, end)| (end.pool, position, end.id.as_str()))
            .collect();
        let losses: Vec<f64> = ends.iter().map(|e| e.entrance_loss_coefficient).collect();
        Ok(Some(Self {
            reach_ends,
            ended,
            pools: index(&ends.iter().map(|e| e.pool).collect::<Vec<_>>())?,
            cuts: index(&pools.end_cuts)?,
            losses: Tensor::<R>::from_slice(&losses, &[losses.len()], device)?,
            tree: TransferTree::new_grouped(client, pools.ids.len(), ends.len(), &targets)
                .map_err(transfer_error)?,
            ends: ends.to_vec(),
        }))
    }
}

impl<R: Runtime<DType = DType>> PoolEndTopology<R> {
    /// True when a pool end owns either end of setup reach `reach`.
    pub fn owns(&self, reach: usize) -> bool {
        self.reach_ends
            .get(reach)
            .is_some_and(|ends| ends.iter().any(Option::is_some))
    }
}
