use super::super::{ChannelError, ChannelLateralDirection, ChannelReachDefinition};
use crate::numerics::outward::AdjustmentReduction;
use crate::numerics::water_transfers::TransferTree;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug)]
pub(in crate::channel::flow) struct LateralContext<R: Runtime<DType = DType>> {
    pub definition: Arc<ChannelReachDefinition>,
    pub cell_indices: Tensor<R>,
    pub inflow_indices: Tensor<R>,
    pub inflow_mask: Tensor<R>,
    pub tree: TransferTree<R>,
    pub cell_reduction: AdjustmentReduction,
    pub binding_reduction: AdjustmentReduction,
}

impl<R: Runtime<DType = DType>> LateralContext<R> {
    pub fn count(&self) -> usize {
        self.definition.lateral_bindings.len()
    }
    pub fn cells(&self) -> usize {
        self.definition.geometry.len()
    }
}

impl<R: Runtime<DType = DType>> LateralContext<R>
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
    pub fn new(
        client: &R::Client,
        definition: Arc<ChannelReachDefinition>,
    ) -> Result<Self, ChannelError> {
        let count = definition.lateral_bindings.len();
        let cells = definition.geometry.len();
        let group_count = cells.checked_mul(2).ok_or_else(|| {
            ChannelError::input("lateral group count overflows: reduce the reach size")
        })?;
        let lookup: BTreeMap<_, _> = definition
            .geometry
            .cells()
            .enumerate()
            .map(|(i, c)| (c.id, i))
            .collect();
        let mut cell_indices = Vec::with_capacity(count);
        let mut inflow_indices = Vec::new();
        let mut masks = Vec::with_capacity(count);
        let mut targets = Vec::with_capacity(count);
        for (index, binding) in definition.lateral_bindings.iter().enumerate() {
            let cell = *lookup.get(binding.cell_id.as_str()).ok_or_else(|| {
                ChannelError::input(format!(
                    "cell_id={:?}: attach lateral input to an existing cell",
                    binding.cell_id
                ))
            })?;
            cell_indices.push(
                i64::try_from(cell)
                    .map_err(|_| ChannelError::input("lateral cell index exceeds I64 support"))?,
            );
            let inflow = binding.direction == ChannelLateralDirection::Inflow;
            if inflow {
                inflow_indices.push(i64::try_from(index).map_err(|_| {
                    ChannelError::input("lateral binding index exceeds I64 support")
                })?);
            }
            masks.push(if inflow { 1.0 } else { 0.0 });
            targets.push((
                if inflow { cell } else { cells + cell },
                index,
                binding.boundary_id.as_str(),
            ));
        }
        let tree = TransferTree::new_grouped(client, group_count, count, &targets)?;
        let reduction =
            |n| AdjustmentReduction::new(n).map_err(|e| ChannelError::numerical(e.to_string()));
        Ok(Self {
            cell_indices: Tensor::from_slice(&cell_indices, &[count], client.device())?,
            inflow_indices: Tensor::from_slice(
                &inflow_indices,
                &[inflow_indices.len()],
                client.device(),
            )?,
            inflow_mask: Tensor::from_slice(&masks, &[count], client.device())?,
            tree,
            cell_reduction: reduction(cells)?,
            binding_reduction: reduction(count)?,
            definition,
        })
    }
}
