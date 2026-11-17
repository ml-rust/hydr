use super::{
    CoupledLevelPoolError,
    context::Context,
    topology::{finite, reject},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

#[derive(Clone, Debug)]
pub(super) struct Heads<R: Runtime<DType = DType>> {
    pub a: Tensor<R>,
    pub b: Tensor<R>,
    pub wet_a: Tensor<R>,
    pub wet_b: Tensor<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
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
        Ok(self.client.gt(volume, &self.zero)?)
    }
    pub fn heads(
        &self,
        stages: &Tensor<R>,
        wet: &Tensor<R>,
    ) -> Result<Heads<R>, CoupledLevelPoolError> {
        let c = &self.client;
        let a = c.index_select(stages, 0, &self.a)?;
        let b = c.index_select(stages, 0, &self.b)?;
        let wa = c.index_select(wet, 0, &self.a)?;
        let wb = c.index_select(wet, 0, &self.b)?;
        Ok(Heads {
            wet_a: c.mul(&wa, &c.gt(&a, &self.bed_a)?)?,
            wet_b: c.mul(&wb, &c.gt(&b, &self.bed_b)?)?,
            a,
            b,
        })
    }
    pub fn midpoint_heads(
        &self,
        start: &Tensor<R>,
        end: &Tensor<R>,
        start_wet: &Tensor<R>,
        end_wet: &Tensor<R>,
    ) -> Result<Heads<R>, CoupledLevelPoolError> {
        let c = &self.client;
        let stage = c.midpoint(start, end)?;
        finite(c, &stage, "temporal midpoint stage")?;
        let wet = c.maximum(start_wet, end_wet)?;
        reject(
            c,
            &c.mul(&wet, &c.le(&stage, &self.origin)?)?,
            "positive pool water has no represented midpoint depth. Rescale the vertical reference",
        )?;
        self.heads(&stage, &wet)
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
