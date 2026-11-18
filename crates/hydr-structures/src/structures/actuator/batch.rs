//! Immutable accepted ideal-actuator batches and positive-duration trials.

use crate::structures::tensor::actuator::{
    TensorActuatorContext, TensorActuatorState, TensorActuatorStep,
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
use crate::structures::{StructureBatch, StructureElement, StructureError};

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
    /// Next strictly later target, arrival window, or terminal event of a coordinator.
    ///
    /// Each certified arrival window `[lower, upper]` encloses one arrival instant, so it
    /// counts as one event. A window ahead of the accepted time reports its upper bound. A
    /// window the accepted time lies inside reports nothing more. `prepare_coordinated_to`
    /// admits every end up to this event. Terminal state returns None.
    #[doc(hidden)]
    pub fn next_coordinated_event(&self) -> Result<Option<GateActuatorEvent>, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>,
    {
        crate::structures::numerics::actuator::next_coordinated_event(
            &self.context,
            &self.state,
            self.time,
            self.next_target,
        )
    }
    /// The event exactly at `end_s`, or `None` when `end_s` holds no target, pending arrival
    /// bound, or terminal horizon.
    pub(crate) fn event_at(&self, end_s: f64) -> Result<Option<GateActuatorEvent>, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>,
    {
        crate::structures::numerics::actuator::event_at(
            &self.context,
            &self.state,
            (self.time, self.next_target),
            end_s,
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
        self.complete(self.prepare_to(end_s)?, target_updates)
    }
}
