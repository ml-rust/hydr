use super::ScalarBoundaryFrame;
use crate::channel::flow::{ChannelBoundary, ChannelError, context::Context};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ShapeOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

pub(in crate::channel::flow) fn is_scalar(boundary: &ChannelBoundary) -> bool {
    boundary.is_scalar()
}

pub(in crate::channel::flow) fn numerical_range(
    count: usize,
    left: &ChannelBoundary,
    right: &ChannelBoundary,
) -> (usize, usize) {
    let start = usize::from(is_scalar(left));
    (start, count + 1 - start - usize::from(is_scalar(right)))
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client:
        BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ReduceOps<R> + ShapeOps<R> + UnaryOps<R>,
{
    pub(in crate::channel::flow) fn endpoint_pair(
        &self,
        left: &Tensor<R>,
        right: &Tensor<R>,
    ) -> Result<Tensor<R>, ChannelError> {
        Ok(self.client.cat(
            &[
                &left.narrow(0, 0, 1)?,
                &right.narrow(0, right.numel() - 1, 1)?,
            ],
            0,
        )?)
    }

    pub(in crate::channel::flow) fn prescribed_water_mask(
        &self,
        frame: Option<&ScalarBoundaryFrame<R>>,
        numerical: Option<&Tensor<R>>,
        left: &ChannelBoundary,
        right: &ChannelBoundary,
    ) -> Result<Option<Tensor<R>>, ChannelError> {
        let Some(frame) = frame else {
            return Ok(None);
        };
        let boundaries = [left, right];
        let values: Vec<_> = frame
            .traces
            .endpoints
            .iter()
            .map(|endpoint| {
                if matches!(boundaries[*endpoint], ChannelBoundary::Discharge { .. }) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let scalar = self.tensor(&values)?;
        let numerical = numerical
            .map(|value| self.constant(value, 0.0))
            .transpose()?;
        Ok(Some(self.assemble_boundary_faces(
            Some(frame),
            numerical.as_ref(),
            Some(&scalar),
        )?))
    }

    pub(in crate::channel::flow) fn assemble_boundary_faces(
        &self,
        frame: Option<&ScalarBoundaryFrame<R>>,
        numerical: Option<&Tensor<R>>,
        scalar: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>, ChannelError> {
        let Some(frame) = frame else {
            return numerical.cloned().ok_or_else(|| {
                ChannelError::numerical("numerical face array absent: inspect boundary assembly")
            });
        };
        let scalar = scalar.ok_or_else(|| {
            ChannelError::numerical("scalar face array absent: inspect boundary assembly")
        })?;
        let left = if frame.traces.endpoints.first() == Some(&0) {
            Some(scalar.narrow(0, 0, 1)?)
        } else {
            None
        };
        let right = if frame.traces.endpoints.last() == Some(&1) {
            Some(scalar.narrow(0, scalar.numel() - 1, 1)?)
        } else {
            None
        };
        let pieces: Vec<_> = left.iter().chain(numerical).chain(right.iter()).collect();
        Ok(self.client.cat(&pieces, 0)?)
    }
}
