use super::{
    CoupledLevelPoolError,
    context::Context,
    topology::{finite, reject},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, IndexingOps,
    MidpointOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

use crate::coupling::links::LinkError;
pub(super) use crate::coupling::links::pool_heads::Heads;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>,
{
    pub fn wet(&self, volume: &Tensor<R>) -> Result<Tensor<R>, CoupledLevelPoolError> {
        self.attachments()
            .wet(&self.client, volume)
            .map_err(LinkError::level_pool)
    }
    pub fn heads(
        &self,
        stages: &Tensor<R>,
        wet: &Tensor<R>,
    ) -> Result<Heads<R>, CoupledLevelPoolError> {
        self.attachments()
            .heads(&self.client, stages, wet)
            .map_err(LinkError::level_pool)
    }
    pub fn midpoint_heads(
        &self,
        start: &Tensor<R>,
        end: &Tensor<R>,
        start_wet: &Tensor<R>,
        end_wet: &Tensor<R>,
    ) -> Result<Heads<R>, CoupledLevelPoolError> {
        self.attachments()
            .midpoint_heads(&self.client, start, end, start_wet, end_wet)
            .map_err(LinkError::level_pool)
    }
    pub fn packet(
        &self,
        end: &Tensor<R>,
        volume: &Tensor<R>,
        pool_wet: &Tensor<R>,
        heads: &Heads<R>,
    ) -> Result<Tensor<R>, CoupledLevelPoolError> {
        Ok(self
            .client
            .cat(&[end, volume, pool_wet, &heads.wet_a, &heads.wet_b], 0)?)
    }
    pub fn check_packet(&self, x: &Tensor<R>) -> Result<(), CoupledLevelPoolError> {
        use numr::runtime::Device;
        let n = self.pool_count();
        let e = self.edge_count();
        if x.shape() != [2 * n + 3 * e]
            || x.dtype() != DType::F64
            || !x.device().is_same(self.client.device())
        {
            return Err(CoupledLevelPoolError::input(
                "iteration packet requires original F64 shape and runtime device",
            ));
        }
        finite(&self.client, x, "iteration packet")?;
        let flags = x.narrow(0, n + e, n + 2 * e)?;
        let c = &self.client;
        reject(
            c,
            &c.sub(
                &self.one,
                &c.maximum(&c.eq(&flags, &self.zero)?, &c.eq(&flags, &self.one)?)?,
            )?,
            "iteration support flags must be exactly zero or one",
        )?;
        Ok(())
    }
}
