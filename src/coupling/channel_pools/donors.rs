use super::{ChannelPoolError, context_types::Context};
use crate::numerics::water_transfers::{AggregationKind, finite, reject, upward_add};
use crate::storage::LevelPoolBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
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
        let c = &self.client;
        let zero = Tensor::<R>::zeros(&[1], DType::F64, c.device())?;
        let outgoing = c.maximum(&c.neg(signed)?, &zero)?;
        let blocks = self.topology.cut_aggregation.blocks.aggregate(
            c,
            &outgoing,
            AggregationKind::Momentum,
        )?;
        let sums = self.topology.cut_aggregation.covers.aggregate(
            c,
            &blocks.value,
            AggregationKind::Momentum,
        )?;
        let errors = self.topology.cut_aggregation.covers.aggregate(
            c,
            &blocks.bound,
            AggregationKind::Momentum,
        )?;
        let bound = upward_add(
            c,
            &sums.bound,
            &upward_add(c, &errors.value, &errors.bound)?,
        )?;
        Ok(upward_add(c, &sums.value, &bound)?)
    }
    pub(super) fn check_cuts(
        &self,
        available: &Tensor<R>,
        demand: &Tensor<R>,
    ) -> Result<(), ChannelPoolError> {
        finite(&self.client, available, "original cut availability")?;
        finite(&self.client, demand, "cut export upper bound")?;
        reject(
            &self.client,
            &self.client.gt(demand, available)?,
            "connection exports exceed original water above an attachment cut. Shorten the complete interval",
        )?;
        Ok(())
    }
    pub(super) fn attachment_wet(
        &self,
        pools: &LevelPoolBatch<R>,
    ) -> Result<Tensor<R>, ChannelPoolError> {
        let c = &self.client;
        let volume = c.index_select(
            pools.volumes_tensor(),
            0,
            &self.topology.connection_pool_indices,
        )?;
        let cut_upper = c.index_select(
            self.cut_plan.upper_volume_m3(),
            0,
            &self.topology.connection_cut_indices,
        )?;
        // Nominal interpolation can round a below-bed stage above the attachment.
        Ok(c.gt(&volume, &cut_upper)?)
    }
    pub(super) fn check_support(
        &self,
        start: &LevelPoolBatch<R>,
        end: &LevelPoolBatch<R>,
        signed: &Tensor<R>,
    ) -> Result<(), ChannelPoolError> {
        let c = &self.client;
        let zero = Tensor::<R>::zeros(&[1], DType::F64, c.device())?;
        let one = Tensor::<R>::full_scalar(&[1], DType::F64, 1.0, c.device())?;
        let active = c.gt(&c.abs(signed)?, &zero)?;
        let supported = c.mul(&self.attachment_wet(start)?, &self.attachment_wet(end)?)?;
        reject(
            c,
            &c.mul(&active, &c.sub(&one, &supported)?)?,
            "active connection loses wet support during the common interval. Shorten the complete interval",
        )?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "donors_tests.rs"]
mod tests;
