//! The coordinator's side of its operating children: interval ends, attempt operations, and
//! completion with the candidate.

use super::{AttemptOperation, CandidateNodes, OperatingChildren, StepOperation};
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::batch::SurfaceChannelBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};
use std::sync::Arc;

/// Next operating children and the interval operation of an accepted candidate.
pub(in super::super) type Completed<R> =
    (Option<OperatingChildren<R>>, Option<Arc<StepOperation<R>>>);

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    /// The trial end `limited`, shortened to the next operating event after the accepted time.
    ///
    /// No interval then contains an event. A retry only shortens it, so no retry crosses one.
    pub(in super::super) fn operating_end(&self, limited: f64) -> Result<f64, SurfaceChannelError> {
        match &self.operating {
            Some(children) => Ok(limited.min(children.next_event_after(self.time_s)?)),
            None => Ok(limited),
        }
    }

    /// Link laws and prepared operating children of the attempt ending at `end`.
    pub(in super::super) fn operation_at(
        &self,
        end: f64,
    ) -> Result<AttemptOperation<R>, SurfaceChannelError> {
        match &self.operating {
            Some(children) => children.prepare(&self.context, self.time_s, end),
            None => Ok(AttemptOperation::declared(&self.context, end)),
        }
    }

    /// Completes the operating children of `operation` at `end` from the candidate `nodes`:
    /// its end stages and end wet masks at every donor group.
    ///
    /// The trial calls it once the candidate passes every other check, so a retryable rejection
    /// still shortens the interval before any decision samples it.
    pub(in super::super) fn complete_operating(
        &self,
        operation: AttemptOperation<R>,
        end: f64,
        nodes: Option<&(Tensor<R>, Tensor<R>)>,
    ) -> Result<Completed<R>, SurfaceChannelError> {
        match (&self.operating, operation.prepared) {
            (Some(children), Some(prepared)) => {
                let candidate = nodes.map(|(stage, wet)| CandidateNodes { stage, wet });
                let points = &operation.operations.points;
                let (next, step) = children.complete(prepared, end, candidate, points)?;
                Ok((Some(next), Some(Arc::new(step))))
            }
            (None, None) => Ok((None, None)),
            _ => Err(SurfaceChannelError::Numerical(
                "the attempt's operation differs from the operating children".into(),
            )),
        }
    }
}
