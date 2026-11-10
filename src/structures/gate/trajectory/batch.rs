//! Immutable prescribed gate evaluation on the complete original structure context.

use super::{
    GateTrajectoryAssignment, GateTrajectoryEvaluation, GateTrajectoryEvent,
    GateTrajectoryInterval, GateTrajectorySupport,
};
use super::{binding::GateTrajectoryData, evaluation::GateTrajectoryEvaluationData};
use crate::cpu::structures::CpuStructureInput;
use crate::structures::{
    StructureBatch, StructureElement, StructureError, StructureHeads, StructureInputErrorKind,
    StructureOperation, StructureProposal,
};
use std::sync::Arc;

/// Persistent prescribed trajectories bound to complete original structure definitions.
#[derive(Clone, Debug)]
pub struct GateTrajectoryBatch {
    pub(crate) inner: Arc<GateTrajectoryData>,
}
impl StructureBatch {
    /// Bind every gate exactly once in its original subsequence order.
    pub fn with_gate_trajectories(
        &self,
        assignments: Vec<GateTrajectoryAssignment>,
    ) -> Result<GateTrajectoryBatch, StructureError> {
        Ok(GateTrajectoryBatch {
            inner: Arc::new(GateTrajectoryData::new(
                Arc::clone(&self.context),
                assignments,
            )?),
        })
    }
}
impl GateTrajectoryBatch {
    /// Complete original law definitions.
    pub fn elements(&self) -> &[StructureElement] {
        &self.inner.context.elements
    }
    /// Complete immutable original gate definitions.
    pub fn assignments(&self) -> &[GateTrajectoryAssignment] {
        &self.inner.assignments
    }
    /// Closed common support without extrapolation.
    pub fn support(&self) -> GateTrajectorySupport {
        self.inner.support
    }
    /// Evaluate immutable point tensors with original support and explicit receipt admission.
    pub fn evaluate(&self, time_s: f64) -> Result<GateTrajectoryEvaluation, StructureError> {
        super::events::check_query(self.inner.support, time_s)?;
        let (opening, absolute_error, input) =
            crate::numerics::structures::gate::trajectory::evaluate(
                &self.inner.context,
                &self.inner.tensors,
                &self.inner.assignments,
                &self.inner.slots,
                time_s,
            )?;
        Ok(GateTrajectoryEvaluation {
            inner: Arc::new(GateTrajectoryEvaluationData {
                binding: Arc::clone(&self.inner),
                time_s,
                opening,
                absolute_error,
                metadata: input.metadata,
            }),
        })
    }
    /// Propose instantaneous flow. Held volumes retain this sampled opening without trajectory integration.
    pub fn propose_at(
        &self,
        time_s: f64,
        heads: &[StructureHeads],
        other_operations: &[StructureOperation],
    ) -> Result<StructureProposal, StructureError> {
        let evaluation = self.evaluate(time_s)?;
        let input = CpuStructureInput::with_trajectory(
            &self.inner.context,
            heads,
            &evaluation.inner,
            other_operations,
        )?;
        let passive = crate::numerics::structures::evaluate(&self.inner.context, &input)?;
        let result =
            crate::numerics::structures::pump::evaluate(&self.inner.context, &input, passive)?;
        let result =
            crate::numerics::structures::gate::evaluate(&self.inner.context, &input, result)?;
        Ok(StructureProposal::from_trajectory(
            Arc::clone(&self.inner.context),
            heads.to_vec(),
            evaluation.inner,
            other_operations.to_vec(),
            result,
        ))
    }
    /// Return the strictly later original event. Terminal support has no later event.
    pub fn next_event_after(
        &self,
        time_s: f64,
    ) -> Result<Option<GateTrajectoryEvent>, StructureError> {
        super::events::check_query(self.inner.support, time_s)?;
        let index = self
            .inner
            .events
            .partition_point(|event| event.time_s <= time_s);
        Ok(self.inner.events.get(index).cloned())
    }
    /// Check complete support and report original events without advancing or integrating.
    pub fn check_interval(
        &self,
        start_s: f64,
        end_s: f64,
    ) -> Result<GateTrajectoryInterval, StructureError> {
        let duration = end_s - start_s;
        if !start_s.is_finite() || !end_s.is_finite() || !duration.is_finite() || duration <= 0.0 {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!(
                    "trajectory interval=[{start_s},{end_s}],duration={duration}: supply finite positive original duration"
                ),
            ));
        }
        super::events::check_query(self.inner.support, start_s)?;
        super::events::check_query(self.inner.support, end_s)?;
        let first = self
            .inner
            .events
            .partition_point(|event| event.time_s <= start_s);
        let end = self
            .inner
            .events
            .partition_point(|event| event.time_s < end_s);
        let end_event = self
            .inner
            .events
            .get(end)
            .filter(|event| event.time_s == end_s)
            .cloned();
        Ok(GateTrajectoryInterval {
            start_s,
            end_s,
            interior_events: self.inner.events[first..end].to_vec(),
            end_event,
        })
    }
}
