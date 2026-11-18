use super::{ChannelPoolError, context_types::Context};
use crate::coupling::links::{LinkError, pool_ends};
use crate::storage::LevelPoolBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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
        + UnaryOps<R>,
{
    pub(super) fn cut_demand(&self, signed: &Tensor<R>) -> Result<Tensor<R>, ChannelPoolError> {
        let cuts = &self.topology.cut_aggregation;
        pool_ends::cut_demand(&self.client, &cuts.blocks, &cuts.covers, signed)
            .map_err(LinkError::channel_pool)
    }
    pub(super) fn check_cuts(
        &self,
        available: &Tensor<R>,
        demand: &Tensor<R>,
    ) -> Result<(), ChannelPoolError> {
        pool_ends::check_cuts(&self.client, available, demand).map_err(LinkError::channel_pool)
    }
    pub(super) fn attachment_wet(
        &self,
        pools: &LevelPoolBatch<R>,
    ) -> Result<Tensor<R>, ChannelPoolError> {
        pool_ends::attachment_wet(
            &self.client,
            pools.volumes_tensor(),
            self.cut_plan.upper_volume_m3(),
            &self.topology.connection_pool_indices,
            &self.topology.connection_cut_indices,
        )
        .map_err(LinkError::channel_pool)
    }
    pub(super) fn check_support(
        &self,
        start: &LevelPoolBatch<R>,
        end: &LevelPoolBatch<R>,
        signed: &Tensor<R>,
    ) -> Result<(), ChannelPoolError> {
        pool_ends::check_support(
            &self.client,
            &self.attachment_wet(start)?,
            &self.attachment_wet(end)?,
            signed,
        )
        .map_err(LinkError::channel_pool)
    }
}

#[cfg(test)]
#[path = "donors_tests.rs"]
mod tests;
