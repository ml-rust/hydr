use crate::channel::flow::{ChannelError, conserved_update::expanded, context::Context};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps, UnaryOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Upward-rounded width of each accepted joint flux bound.
    pub(super) fn surveyed_width(&self, bound: &Interval<R>) -> Result<Tensor<R>, ChannelError> {
        expanded(&self.client, "surveyed reach flux width", |a| {
            let lower = a.point(bound.lower())?;
            let upper = a.point(bound.upper())?;
            Ok(a.sub(&upper, &lower)?.upper().clone())
        })
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
