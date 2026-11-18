use super::DirectBoundaryFrame;
use crate::channel::flow::{ChannelError, ChannelTensorBoundary, context::Context};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ShapeOps, UnaryOps},
    runtime::Runtime,
    tensor::Tensor,
};

pub(in crate::channel::flow) fn is_scalar<R: Runtime<DType = DType>>(
    boundary: &ChannelTensorBoundary<R>,
) -> bool {
    boundary.is_scalar()
}

/// True when the end face takes its rates from the policy.
pub(in crate::channel::flow) fn is_direct<R: Runtime<DType = DType>>(
    boundary: &ChannelTensorBoundary<R>,
) -> bool {
    boundary.is_direct()
}

/// First face and face count that a face Riemann problem solves.
pub(in crate::channel::flow) fn numerical_range<R: Runtime<DType = DType>>(
    count: usize,
    left: &ChannelTensorBoundary<R>,
    right: &ChannelTensorBoundary<R>,
) -> (usize, usize) {
    let start = usize::from(is_direct(left));
    (start, count + 1 - start - usize::from(is_direct(right)))
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
        frame: Option<&DirectBoundaryFrame<R>>,
        numerical: Option<&Tensor<R>>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<Option<Tensor<R>>, ChannelError> {
        let Some(frame) = frame else {
            return Ok(None);
        };
        let boundaries = [left, right];
        let values: Vec<_> = frame
            .endpoints
            .iter()
            .map(|endpoint| {
                if matches!(
                    boundaries[*endpoint],
                    ChannelTensorBoundary::Discharge { .. }
                ) {
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

    /// Places one entry per direct endpoint around the numerical face entries.
    pub(in crate::channel::flow) fn assemble_boundary_faces(
        &self,
        frame: Option<&DirectBoundaryFrame<R>>,
        numerical: Option<&Tensor<R>>,
        direct: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>, ChannelError> {
        let Some(frame) = frame else {
            return numerical.cloned().ok_or_else(|| {
                ChannelError::numerical("numerical face array absent: inspect boundary assembly")
            });
        };
        let direct = direct.ok_or_else(|| {
            ChannelError::numerical("direct face array absent: inspect boundary assembly")
        })?;
        let left = if frame.endpoints.first() == Some(&0) {
            Some(direct.narrow(0, 0, 1)?)
        } else {
            None
        };
        let right = if frame.endpoints.last() == Some(&1) {
            Some(direct.narrow(0, direct.numel() - 1, 1)?)
        } else {
            None
        };
        let pieces: Vec<_> = left.iter().chain(numerical).chain(right.iter()).collect();
        Ok(self.client.cat(&pieces, 0)?)
    }
}
