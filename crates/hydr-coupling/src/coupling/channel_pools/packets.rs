use super::{ChannelPoolError, context_types::Context};
use crate::numerics::water_transfers::{finite, reject};
use crate::storage::LevelPoolBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Device;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{AndersonAlgorithms, BatchedBisectionAlgorithms};

pub(super) struct Residuals<R: Runtime<DType = DType>> {
    pub head: Tensor<R>,
    pub transfer: Tensor<R>,
    pub norm: f64,
    pub converged: bool,
}
impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>
        + BatchedBisectionAlgorithms<R>
        + AndersonAlgorithms<R>,
{
    pub(super) fn packet(
        &self,
        pools: &LevelPoolBatch<R>,
        volumes: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelPoolError> {
        let c = &self.client;
        let zero = Tensor::<R>::zeros(&[1], DType::F64, c.device())?;
        Ok(c.cat(
            &[
                &pools.stages_tensor()?,
                volumes,
                &c.gt(pools.volumes_tensor(), &zero)?,
                &self.attachment_wet(pools)?,
            ],
            0,
        )?)
    }
    pub(super) fn initial_packet(
        &self,
        pools: &LevelPoolBatch<R>,
    ) -> Result<Tensor<R>, ChannelPoolError> {
        self.packet(
            pools,
            &Tensor::<R>::zeros(
                &[self.setup.connections.len()],
                DType::F64,
                self.client.device(),
            )?,
        )
    }
    pub(super) fn check_packet(&self, x: &Tensor<R>) -> Result<(), ChannelPoolError> {
        let p = self.setup.pools.len();
        let e = self.setup.connections.len();
        if x.shape() != [2 * p + 2 * e]
            || x.dtype() != DType::F64
            || !x.device().is_same(self.client.device())
        {
            return Err(ChannelPoolError::input(
                "iteration packet shape, dtype, or device differs: supply the complete native packet",
            ));
        }
        finite(&self.client, x, "channel/pool iteration packet")
            .map_err(ChannelPoolError::from_transfer)?;
        let flags = x.narrow(0, p + e, p + e)?;
        let zero = Tensor::<R>::zeros(&[1], DType::F64, self.client.device())?;
        let one = Tensor::<R>::full_scalar(&[1], DType::F64, 1.0, self.client.device())?;
        let valid = self.client.maximum(
            &self.client.eq(&flags, &zero)?,
            &self.client.eq(&flags, &one)?,
        )?;
        reject(
            &self.client,
            &self.client.sub(&one, &valid)?,
            "iteration support flags differ from zero or one",
        )
        .map_err(ChannelPoolError::from_transfer)?;
        Ok(())
    }
    pub(super) fn residuals(
        &self,
        x: &Tensor<R>,
        gx: &Tensor<R>,
    ) -> Result<Residuals<R>, ChannelPoolError> {
        self.check_packet(x)?;
        self.check_packet(gx)?;
        let c = &self.client;
        let p = self.setup.pools.len();
        let e = self.setup.connections.len();
        let difference = c.abs(&c.sub(gx, x)?)?;
        finite(c, &difference, "fixed-point residual").map_err(ChannelPoolError::from_transfer)?;
        let heads = difference.narrow(0, 0, p)?;
        let transfers = difference.narrow(0, p, e)?;
        let head_scale = c.maximum(
            &c.abs(&c.sub(&x.narrow(0, 0, p)?, &self.topology.pool_origins_m)?)?,
            &c.abs(&c.sub(&gx.narrow(0, 0, p)?, &self.topology.pool_origins_m)?)?,
        )?;
        let transfer_scale =
            c.maximum(&c.abs(&x.narrow(0, p, e)?)?, &c.abs(&gx.narrow(0, p, e)?)?)?;
        let config = &self.setup.config;
        let head_budget = c.add_scalar(
            &c.mul_scalar(&head_scale, config.head_relative)?,
            config.head_absolute_m,
        )?;
        let transfer_budget = c.add_scalar(
            &c.mul_scalar(&transfer_scale, config.transfer_relative)?,
            config.transfer_absolute_m3,
        )?;
        finite(c, &head_budget, "head convergence budget")
            .map_err(ChannelPoolError::from_transfer)?;
        finite(c, &transfer_budget, "transfer convergence budget")
            .map_err(ChannelPoolError::from_transfer)?;
        let zero = Tensor::<R>::zeros(&[1], DType::F64, c.device())?;
        let one = Tensor::<R>::full_scalar(&[1], DType::F64, 1.0, c.device())?;
        let normalized =
            |residual: &Tensor<R>, budget: &Tensor<R>| -> Result<Tensor<R>, ChannelPoolError> {
                let positive = c.gt(budget, &zero)?;
                let safe = c.where_cond(&positive, budget, &one)?;
                let ratio = c.div(residual, &safe)?;
                let fallback = c.where_cond(
                    &c.eq(residual, &zero)?,
                    &zero,
                    &Tensor::<R>::full_scalar(&[1], DType::F64, f64::INFINITY, c.device())?,
                )?;
                Ok(c.where_cond(&positive, &ratio, &fallback)?)
            };
        let norm = c
            .maximum(
                &c.max(&normalized(&heads, &head_budget)?, &[0], false)?,
                &c.max(&normalized(&transfers, &transfer_budget)?, &[0], false)?,
            )?
            .item::<f64>()?;
        let flags = c
            .max(&difference.narrow(0, p + e, p + e)?, &[0], false)?
            .item::<f64>()?
            == 0.0;
        Ok(Residuals {
            head: c.max(&heads, &[0], true)?,
            transfer: c.max(&transfers, &[0], true)?,
            norm,
            converged: flags && norm <= 1.0,
        })
    }
}
