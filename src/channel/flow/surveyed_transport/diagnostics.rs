use super::error::riemann_error;
use crate::channel::flow::{
    ChannelError,
    context::Context,
    surveyed::{
        PrismaticRiemannWork,
        error::{checked, into_public},
    },
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
        UnaryOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + MidpointOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn surveyed_nominal(
        &self,
        bounds: &Interval<R>,
        work: PrismaticRiemannWork,
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let c = &self.client;
        let (value, width) = checked::<R, _>(c, "surveyed reach represented flux", |a| {
            let value = c.midpoint(bounds.lower(), bounds.upper())?;
            let _ = a.point(&value)?;
            let lower = a.point(bounds.lower())?;
            let upper = a.point(bounds.upper())?;
            let width = a.sub(&upper, &lower)?.upper().clone();
            Ok((value, width))
        })
        .map_err(|e| riemann_error(into_public(e, work)))?;
        let inside = c.mul(
            &c.ge(&value, bounds.lower())?,
            &c.le(&value, bounds.upper())?,
        )?;
        if !self.all(&inside)? {
            return Err(ChannelError::numerical(
                "surveyed represented midpoint escapes its flux certificate: inspect native midpoint arithmetic",
            ));
        }
        Ok((value, width))
    }

    pub(super) fn surveyed_wave_envelope(
        &self,
        bounds: &Interval<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        let speed = self.client.maximum(
            &self.client.abs(bounds.lower())?,
            &self.client.abs(bounds.upper())?,
        )?;
        self.finite(&speed, "surveyed complete wave envelope")?;
        Ok(speed)
    }
}
