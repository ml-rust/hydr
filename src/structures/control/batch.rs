//! Immutable sampled-control trials with explicit ownership-based acceptance.

use super::{
    ControlCommandOutput, ControlDecisionSchedule, ControlEvent, ControlInitialState,
    ControlSample, ControlState, ControlStepDiagnostics, OperatingControlDefinition,
    OperatingControlTrial, SampledControlTiming,
};
use crate::structures::{
    StructureBatch, StructureElement, StructureError, StructureInputErrorKind,
};
use crate::tensor::structures::control::{
    TensorControlContext, TensorControlState, TensorControlStep, TensorControlTimer,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Complete sampled command continuation without hydraulic or actuator advancement.
#[derive(Clone, Debug)]
pub struct OperatingControlBatch<R: Runtime<DType = DType>> {
    pub(crate) context: Arc<TensorControlContext<R>>,
    pub(crate) state: TensorControlState<R>,
    pub(crate) time: f64,
    pub(crate) next_decision: usize,
    pub(crate) steps: usize,
    pub(crate) last: Option<Arc<TensorControlStep<R>>>,
}
impl<R: Runtime<DType = DType>> StructureBatch<R> {
    /// Bind explicit sampled relay definitions to actual devices and complete initial memory.
    pub fn with_operating_controls(
        &self,
        definitions: Vec<OperatingControlDefinition>,
        initial_states: Vec<ControlInitialState>,
        schedule: ControlDecisionSchedule,
        timing: SampledControlTiming,
    ) -> Result<OperatingControlBatch<R>, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let context = Arc::new(TensorControlContext::new(
            Arc::clone(&self.context),
            definitions,
            initial_states,
            schedule,
            timing,
        )?);
        let state = TensorControlState::initial(&context)?;
        Ok(OperatingControlBatch {
            time: context.schedule.initial_time_s,
            context,
            state,
            next_decision: 0,
            steps: 0,
            last: None,
        })
    }
}
impl<R: Runtime<DType = DType>> OperatingControlBatch<R> {
    /// Complete immutable actual structure definitions.
    pub fn elements(&self) -> &[StructureElement] {
        &self.context.structures.elements
    }
    /// Original ordered controlled subset.
    pub fn definitions(&self) -> &[OperatingControlDefinition] {
        &self.context.definitions
    }
    /// Complete original assumed antecedents.
    pub fn initial_states(&self) -> &[ControlInitialState] {
        &self.context.initial_states
    }
    /// Complete original future decision schedule.
    pub fn schedule(&self) -> &ControlDecisionSchedule {
        &self.context.schedule
    }
    /// Original sampled applicability and timing bounds.
    pub fn timing(&self) -> &SampledControlTiming {
        &self.context.timing
    }
    /// Accepted time in seconds.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Accepted original interval count.
    pub fn accepted_steps(&self) -> usize {
        self.steps
    }
    /// Next original decision or terminal support boundary.
    pub fn next_event(&self) -> Option<ControlEvent> {
        if self.time == self.schedule().end_s {
            return None;
        }
        match self.schedule().decision_times_s.get(self.next_decision) {
            Some(&time_s) => Some(ControlEvent {
                time_s,
                decision_index: Some(self.next_decision),
                terminal: time_s == self.schedule().end_s,
            }),
            None => Some(ControlEvent {
                time_s: self.schedule().end_s,
                decision_index: None,
                terminal: true,
            }),
        }
    }
    /// Explicitly project commands without claiming physical device response.
    pub fn commands(&self) -> Result<Vec<ControlCommandOutput>, StructureError> {
        super::projection::commands(&self.context, &self.state, self.time)
    }
    /// Explicitly project every continuation-relevant controller value.
    pub fn states(&self) -> Result<Vec<ControlState>, StructureError> {
        super::projection::states(&self.context, &self.state)
    }
    /// Explicitly project the bounded last accepted interval.
    pub fn last_step(&self) -> Result<Option<ControlStepDiagnostics>, StructureError> {
        self.last
            .as_ref()
            .map(|step| super::projection::step(&self.context, step))
            .transpose()
    }
    /// Trial an interval without crossing an unprocessed decision or modifying accepted state.
    pub fn trial_to(
        &self,
        end_s: f64,
        decision_samples: &[ControlSample],
    ) -> Result<OperatingControlTrial<R>, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let duration = end_s - self.time;
        if !end_s.is_finite() || !duration.is_finite() || duration <= 0.0 {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!(
                    "control interval=[{},{}],duration={duration}: supply a finite positive original duration",
                    self.time, end_s
                ),
            ));
        }
        let event = self.next_event().ok_or_else(|| {
            StructureError::input(
                StructureInputErrorKind::UnsupportedRegime,
                "control trial exceeds terminal support",
            )
        })?;
        if end_s > event.time_s {
            return Err(StructureError::input(
                StructureInputErrorKind::UnsupportedRegime,
                "control interval crosses the next original event: split at the event",
            ));
        }
        let decision = if end_s == event.time_s {
            event.decision_index
        } else {
            None
        };
        let step = match decision {
            Some(index) => {
                let input = crate::tensor::structures::control::input::decision_samples(
                    &self.context,
                    decision_samples,
                    end_s,
                )?;
                crate::numerics::structures::control::decide(
                    &self.context,
                    &self.state,
                    self.time,
                    end_s,
                    index,
                    decision_samples,
                    input,
                )?
            }
            None => {
                if !decision_samples.is_empty() {
                    return Err(StructureError::input(
                        StructureInputErrorKind::OperatingPoint,
                        "non-decision control interval requires an empty sample slice",
                    ));
                }
                TensorControlStep {
                    start: self.time,
                    end: end_s,
                    decision_index: None,
                    before: self.state.clone(),
                    after: self.state.clone(),
                    samples: Vec::new(),
                    qualifies: self.context.zero.clone(),
                    timer: TensorControlTimer::zero(&self.context),
                    transitioned: self.context.zero.clone(),
                }
            }
        };
        let steps = self.steps.checked_add(1).ok_or_else(|| {
            StructureError::numerical("control accepted interval counter overflows")
        })?;
        let next_decision = if decision.is_some() {
            self.next_decision
                .checked_add(1)
                .ok_or_else(|| StructureError::numerical("control decision counter overflows"))?
        } else {
            self.next_decision
        };
        let next = Self {
            context: Arc::clone(&self.context),
            state: step.after.clone(),
            time: end_s,
            next_decision,
            steps,
            last: Some(Arc::new(step)),
        };
        Ok(OperatingControlTrial { next })
    }
}
