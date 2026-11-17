//! Immutable accepted ideal-actuator batches and positive-duration trials.

use crate::structures::tensor::actuator::{
    TensorActuatorContext, TensorActuatorState, TensorActuatorStep, TensorActuatorUpdate,
    record::TensorRecord,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Accepted persistent ideal response state bound to the complete actual structure context.
#[derive(Clone, Debug)]
pub struct GateActuatorBatch<R: Runtime<DType = DType>> {
    pub(crate) context: Arc<TensorActuatorContext<R>>,
    pub(crate) state: TensorActuatorState<R>,
    pub(crate) time: f64,
    pub(crate) next_target: usize,
    pub(crate) steps: usize,
    pub(crate) last: Option<Arc<TensorActuatorStep<R>>>,
}

use super::{
    GateActuatorDefinition, GateActuatorEvent, GateActuatorInitialState, GateActuatorSchedule,
    GateActuatorTrial, GateTargetUpdate,
};
use crate::structures::{
    StructureBatch, StructureElement, StructureError, StructureInputErrorKind,
};

impl<R: Runtime<DType = DType>> StructureBatch<R> {
    /// Bind an explicit ideal finite-rate response to an ordered actual-gate subset.
    pub fn with_gate_actuators(
        &self,
        definitions: Vec<GateActuatorDefinition>,
        initial_states: Vec<GateActuatorInitialState>,
        schedule: GateActuatorSchedule,
    ) -> Result<GateActuatorBatch<R>, StructureError>
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
        let context = Arc::new(TensorActuatorContext::new(
            Arc::clone(&self.context),
            definitions,
            initial_states,
            schedule,
        )?);
        let state = crate::structures::numerics::actuator::initial(&context)?;
        Ok(GateActuatorBatch {
            time: context.schedule.initial_time_s,
            context,
            state,
            next_target: 0,
            steps: 0,
            last: None,
        })
    }
}
impl<R: Runtime<DType = DType>> GateActuatorBatch<R> {
    /// Complete bound original structure definitions in their original order.
    pub fn elements(&self) -> &[StructureElement] {
        &self.context.structures.elements
    }
    /// Immutable ordered actuator definitions and explicit policies.
    pub fn definitions(&self) -> &[GateActuatorDefinition] {
        &self.context.definitions
    }
    /// Original supplied initial positions and targets without bit normalization.
    pub fn initial_states(&self) -> &[GateActuatorInitialState] {
        &self.context.initial_states
    }
    /// Original finite target calendar and supported horizon.
    pub fn schedule(&self) -> &GateActuatorSchedule {
        &self.context.schedule
    }
    /// Accepted response time in seconds.
    pub fn time(&self) -> f64 {
        self.time
    }
    /// Number of accepted positive-duration response intervals.
    pub fn accepted_steps(&self) -> usize {
        self.steps
    }
    /// Next strictly later target, arrival boundary, or terminal event. Terminal state returns None.
    pub fn next_event(&self) -> Result<Option<GateActuatorEvent>, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>,
    {
        crate::structures::numerics::actuator::next_event(
            &self.context,
            &self.state,
            self.time,
            self.next_target,
        )
    }
    /// Produce a complete candidate without crossing any original command or arrival event.
    pub fn trial_to(
        &self,
        end_s: f64,
        target_updates: &[GateTargetUpdate],
    ) -> Result<GateActuatorTrial<R>, StructureError>
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
        self.trial_admitting(end_s, |context, expected| {
            crate::structures::numerics::actuator::admit_update(
                context,
                target_updates,
                end_s,
                expected,
            )
        })
    }
    /// Advance with a complete tensor update admitted by the bound command owner.
    pub(crate) fn trial_to_with_update(
        &self,
        end_s: f64,
        update: Option<TensorActuatorUpdate<R>>,
    ) -> Result<GateActuatorTrial<R>, StructureError>
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
        self.trial_admitting(end_s, |context, expected| match (expected, update) {
            (true, Some(update)) => Ok(update),
            (false, None) => Ok(TensorActuatorUpdate::zero(context)),
            (true, None) => Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                "target event requires a complete admitted tensor update",
            )),
            (false, Some(_)) => Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                "non-target actuator interval requires no tensor update",
            )),
        })
    }
    fn trial_admitting(
        &self,
        end_s: f64,
        admit: impl FnOnce(
            &TensorActuatorContext<R>,
            bool,
        ) -> Result<TensorActuatorUpdate<R>, StructureError>,
    ) -> Result<GateActuatorTrial<R>, StructureError>
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
                    "actuator interval=[{},{}],duration={duration}: supply a finite positive original duration",
                    self.time, end_s
                ),
            ));
        }
        let next = self.next_event()?.ok_or_else(|| {
            StructureError::input(
                StructureInputErrorKind::UnsupportedRegime,
                "actuator trial exceeds terminal support",
            )
        })?;
        if end_s > next.time_s {
            return Err(StructureError::input(
                StructureInputErrorKind::UnsupportedRegime,
                "actuator interval crosses the next original event: split at that event",
            ));
        }
        let event = if end_s == next.time_s {
            Some(next)
        } else {
            None
        };
        let target_index = event.and_then(|e| e.target_index);
        let steps = self.steps.checked_add(1).ok_or_else(|| {
            StructureError::numerical("actuator accepted interval counter overflow")
        })?;
        let next_target = if target_index.is_some() {
            self.next_target.checked_add(1).ok_or_else(|| {
                StructureError::numerical("actuator target event counter overflow")
            })?
        } else {
            self.next_target
        };
        let update = admit(&self.context, target_index.is_some())?;
        let step = crate::structures::numerics::actuator::step(
            &self.context,
            &self.state,
            self.time,
            end_s,
            event,
            target_index,
            update,
        )?;
        let next = Self {
            context: Arc::clone(&self.context),
            state: step.after.clone(),
            time: end_s,
            next_target,
            steps,
            last: Some(Arc::new(step)),
        };
        Ok(GateActuatorTrial { next })
    }
}
