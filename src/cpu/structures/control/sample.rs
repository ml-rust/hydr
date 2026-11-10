//! Original sample values retained as persistent tensor state.

use super::CpuControlContext;
use crate::cpu::CpuTensor;
use crate::structures::{ControlSample, StructureError};

#[derive(Clone, Debug)]
pub(crate) struct CpuControlSample {
    pub valid: CpuTensor,
    pub observed: CpuTensor,
    pub available: CpuTensor,
    pub stage: CpuTensor,
}
impl CpuControlSample {
    pub(crate) fn from_optional(
        context: &CpuControlContext,
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
    pub(crate) fn zero(context: &CpuControlContext) -> Self {
        Self {
            valid: context.zero.clone(),
            observed: context.zero.clone(),
            available: context.zero.clone(),
            stage: context.zero.clone(),
        }
    }
}
