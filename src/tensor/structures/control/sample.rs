//! Original sample values retained as persistent tensor state.

use super::TensorControlContext;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::structures::{ControlSample, StructureError};

#[derive(Clone, Debug)]
pub(crate) struct TensorControlSample<R: Runtime<DType = DType>> {
    pub valid: Tensor<R>,
    pub observed: Tensor<R>,
    pub available: Tensor<R>,
    pub stage: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorControlSample<R> {
    pub(crate) fn from_optional(
        context: &TensorControlContext<R>,
        samples: &[Option<&ControlSample>],
    ) -> Result<Self, StructureError> {
        Ok(Self {
            valid: context.tensor(
                &samples
                    .iter()
                    .map(|s| if s.is_some() { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
            )?,
            observed: context.tensor(
                &samples
                    .iter()
                    .map(|s| s.as_ref().map_or(0.0, |s| s.observed_at_s))
                    .collect::<Vec<_>>(),
            )?,
            available: context.tensor(
                &samples
                    .iter()
                    .map(|s| s.as_ref().map_or(0.0, |s| s.available_at_s))
                    .collect::<Vec<_>>(),
            )?,
            stage: context.tensor(
                &samples
                    .iter()
                    .map(|s| s.as_ref().map_or(0.0, |s| s.stage_m))
                    .collect::<Vec<_>>(),
            )?,
        })
    }
    pub(crate) fn zero(context: &TensorControlContext<R>) -> Self {
        Self {
            valid: context.zero.clone(),
            observed: context.zero.clone(),
            available: context.zero.clone(),
            stage: context.zero.clone(),
        }
    }
}
