//! Persistent sampled-control definitions on the complete actual structure context.

use crate::structures::{
    ControlDecisionSchedule, ControlInitialState, OperatingControlDefinition, SampledControlTiming,
    StructureError, StructureInputErrorKind,
};
use crate::tensor::structures::TensorStructureContext;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

pub(crate) const MAX_EXACT_COUNT: u64 = 1u64 << 53;
#[derive(Debug)]
pub(crate) struct TensorControlContext<R: Runtime<DType = DType>> {
    pub structures: Arc<TensorStructureContext<R>>,
    pub definitions: Vec<OperatingControlDefinition>,
    pub initial_states: Vec<ControlInitialState>,
    pub schedule: ControlDecisionSchedule,
    pub timing: SampledControlTiming,
    pub slots: Vec<usize>,
    pub zero: Tensor<R>,
    pub one: Tensor<R>,
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
    pub high_delay: Tensor<R>,
    pub low_delay: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorControlContext<R> {
    pub(crate) fn new(
        structures: Arc<TensorStructureContext<R>>,
        definitions: Vec<OperatingControlDefinition>,
        initial_states: Vec<ControlInitialState>,
        schedule: ControlDecisionSchedule,
        timing: SampledControlTiming,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let slots = super::definition::check(
            &structures,
            &definitions,
            &initial_states,
            &schedule,
            &timing,
        )?;
        let tensor = |values: &[f64]| structures.tensor(values);
        let zero = tensor(&vec![0.0; definitions.len()])?;
        let one = tensor(&vec![1.0; definitions.len()])?;
        let lower = tensor(
            &definitions
                .iter()
                .map(|d| d.lower_stage_m)
                .collect::<Vec<_>>(),
        )?;
        let upper = tensor(
            &definitions
                .iter()
                .map(|d| d.upper_stage_m)
                .collect::<Vec<_>>(),
        )?;
        let high_delay = tensor(
            &definitions
                .iter()
                .map(|d| d.high_qualification_delay_s)
                .collect::<Vec<_>>(),
        )?;
        let low_delay = tensor(
            &definitions
                .iter()
                .map(|d| d.low_qualification_delay_s)
                .collect::<Vec<_>>(),
        )?;
        let context = Self {
            structures,
            definitions,
            initial_states,
            schedule,
            timing,
            slots,
            zero,
            one,
            lower,
            upper,
            high_delay,
            low_delay,
        };
        crate::numerics::structures::control::check_schedule(&context)?;
        Ok(context)
    }
    pub(crate) fn tensor(&self, values: &[f64]) -> Result<Tensor<R>, StructureError> {
        self.structures.tensor(values)
    }
    pub(crate) fn reject(
        &self,
        mask: &Tensor<R>,
        kind: Option<StructureInputErrorKind>,
        message: &str,
    ) -> Result<(), StructureError>
    where
        R::Client: numr::ops::ReduceOps<R>,
    {
        if self.structures.any(mask)? {
            let flags = mask.try_to_vec::<f64>()?;
            let lane = flags.iter().position(|v| *v != 0.0).ok_or_else(|| {
                StructureError::numerical("control rejection mask contradicts reduction")
            })?;
            let slot = *self.slots.get(lane).ok_or_else(|| {
                StructureError::numerical("control rejection mask has incompatible lane count")
            })?;
            return Err(match kind {
                Some(kind) => StructureError::input(kind, message).at(slot),
                None => StructureError::numerical(message).at(slot),
            });
        }
        Ok(())
    }
}
