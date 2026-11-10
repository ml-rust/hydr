//! Persistent sampled-control definitions on the complete actual structure context.

use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::{
    ControlDecisionSchedule, ControlInitialState, OperatingControlDefinition, SampledControlTiming,
    StructureError, StructureInputErrorKind,
};
use std::sync::Arc;

pub(crate) const MAX_EXACT_COUNT: u64 = 1u64 << 53;
#[derive(Debug)]
pub(crate) struct CpuControlContext {
    pub structures: Arc<CpuStructureContext>,
    pub definitions: Vec<OperatingControlDefinition>,
    pub initial_states: Vec<ControlInitialState>,
    pub schedule: ControlDecisionSchedule,
    pub timing: SampledControlTiming,
    pub slots: Vec<usize>,
    pub zero: CpuTensor,
    pub one: CpuTensor,
    pub lower: CpuTensor,
    pub upper: CpuTensor,
    pub high_delay: CpuTensor,
    pub low_delay: CpuTensor,
}
impl CpuControlContext {
    pub(crate) fn new(
        structures: Arc<CpuStructureContext>,
        definitions: Vec<OperatingControlDefinition>,
        initial_states: Vec<ControlInitialState>,
        schedule: ControlDecisionSchedule,
        timing: SampledControlTiming,
    ) -> Result<Self, StructureError> {
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
    pub(crate) fn tensor(&self, values: &[f64]) -> Result<CpuTensor, StructureError> {
        self.structures.tensor(values)
    }
    pub(crate) fn reject(
        &self,
        mask: &CpuTensor,
        kind: Option<StructureInputErrorKind>,
        message: &str,
    ) -> Result<(), StructureError> {
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
