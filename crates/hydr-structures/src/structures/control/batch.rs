//! Immutable sampled-control trials with explicit ownership-based acceptance.

use super::{
    ControlCommandOutput, ControlDecisionSchedule, ControlEvent, ControlInitialState,
    ControlSample, ControlState, ControlStepDiagnostics, OperatingControlDefinition,
    OperatingControlTrial, SampledControlTiming,
};
use crate::structures::tensor::control::{
    TensorControlContext, TensorControlState, TensorControlStep, TensorControlTimer,
};
use crate::structures::{
    StructureBatch, StructureElement, StructureError, StructureInputErrorKind,
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
        self.complete(self.prepare_to(end_s)?, decision_samples)
    }
    /// Check one event-free interval and align its end to the next event without samples.
    pub(crate) fn prepare_to(
        &self,
        end_s: f64,
    ) -> Result<OperatingControlPreparedInterval<R>, StructureError> {
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
        Ok(OperatingControlPreparedInterval {
            base: self.clone(),
            end_s,
            event: (end_s == event.time_s).then_some(event),
        })
    }
    /// Apply the decision samples of a prepared interval on its right side.
    pub(crate) fn complete(
        &self,
        prepared: OperatingControlPreparedInterval<R>,
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
        if !self.same_accepted(&prepared.base) {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                "prepared control interval belongs to another accepted state: prepare it from this batch",
            ));
        }
        let end_s = prepared.end_s;
        let decision = prepared.decision_index();
        let step = match decision {
            Some(index) => {
                let input = crate::structures::tensor::control::input::decision_samples(
                    &self.context,
                    decision_samples,
                    end_s,
                )?;
                crate::structures::numerics::control::decide(
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
    /// Same context, clock, counters, and latch storage as `other`.
    pub(crate) fn same_accepted(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.context, &other.context)
            && self.time.to_bits() == other.time.to_bits()
            && self.steps == other.steps
            && self.next_decision == other.next_decision
            && self.state.high.shares_storage_with(&other.state.high)
            && self.state.pending.shares_storage_with(&other.state.pending)
            && self.state.since.shares_storage_with(&other.state.since)
            && self.state.count.shares_storage_with(&other.state.count)
    }
}

/// One checked event-free control interval before its decision samples.
#[derive(Clone, Debug)]
pub(crate) struct OperatingControlPreparedInterval<R: Runtime<DType = DType>> {
    pub base: OperatingControlBatch<R>,
    pub end_s: f64,
    pub event: Option<ControlEvent>,
}
impl<R: Runtime<DType = DType>> OperatingControlPreparedInterval<R> {
    /// Decision index exactly at the interval end.
    pub fn decision_index(&self) -> Option<usize> {
        self.event.and_then(|event| event.decision_index)
    }
    /// Left-side controls at the interval end: the accepted state on the end clock.
    pub fn left(&self) -> OperatingControlBatch<R> {
        OperatingControlBatch {
            time: self.end_s,
            ..self.base.clone()
        }
    }
}
