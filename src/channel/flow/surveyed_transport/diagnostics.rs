use crate::channel::flow::{
    ChannelError, cfl_speeds::FaceSpeeds, conserved_update::expanded, context::Context,
};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
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

    /// Inward speeds from the complete wave interval: lower is left, upper is right.
    pub(super) fn surveyed_face_speeds(
        &self,
        bounds: &Interval<R>,
    ) -> Result<FaceSpeeds<R>, ChannelError> {
        self.directional_speeds(bounds.lower(), bounds.upper())
    }
}
