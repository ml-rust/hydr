//! Bounded advancement of accepted solver state.

use super::{Solver, StepDiagnostics};
use crate::contract::EngineError;
use crate::model::CellState;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, HypotOps, IndexingOps, ReduceOps,
    ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

/// Why an advancement call returns at its accepted-state boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdvanceStatus {
    /// The requested elapsed time is reached exactly.
    ReachedTarget,
    /// The caller requests cancellation before the next numerical step.
    Cancelled,
    /// The caller's accepted-step budget is exhausted.
    StepLimit,
}

/// Bounded advancement outcome. An incomplete result retains a complete accepted checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct AdvanceReport {
    /// Completion or safe interruption status.
    pub status: AdvanceStatus,
    /// Last accepted elapsed time in seconds.
    pub accepted_time: f64,
    /// Steps accepted during this call.
    pub accepted_steps: usize,
}

impl<R: Runtime<DType = DType>> Solver<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + TypeConversionOps<R>
        + HypotOps<R>
        + numr::ops::CumulativeOps<R>
        + numr::ops::ShapeOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Advances toward a supported time under a bounded accepted-step budget.
    ///
    /// # Errors
    /// Invalid targets reject before advancement. Numerical errors retain the last accepted step.
    pub fn advance_to(
        &mut self,
        target: f64,
        max_steps: usize,
    ) -> Result<AdvanceReport, EngineError> {
        self.advance_inner(
            target,
            max_steps,
            || false,
            None::<fn(&StepDiagnostics, &[CellState])>,
        )
    }

    /// Advances with cancellation and observations only at safe accepted-state boundaries.
    ///
    /// Callbacks own no numerical state. The observer receives each accepted field without interpolation.
    ///
    /// # Errors
    /// Invalid targets or unresolved trials return actionable errors with the last accepted state intact.
    pub fn advance_controlled(
        &mut self,
        target: f64,
        max_steps: usize,
        mut cancelled: impl FnMut() -> bool,
        mut on_accepted: impl FnMut(&StepDiagnostics, &[CellState]),
    ) -> Result<AdvanceReport, EngineError> {
        self.advance_inner(target, max_steps, &mut cancelled, Some(&mut on_accepted))
    }

    fn advance_inner(
        &mut self,
        target: f64,
        max_steps: usize,
        mut cancelled: impl FnMut() -> bool,
        mut observer: Option<impl FnMut(&StepDiagnostics, &[CellState])>,
    ) -> Result<AdvanceReport, EngineError> {
        let end = self
            .definition
            .forcing
            .intervals()
            .last()
            .map(|interval| interval.end);
        if !target.is_finite() || target < self.time() || end.is_none_or(|end| target > end) {
            return Err(EngineError::rejected(
                "target",
                format!(
                    "target={target}, accepted_time={}: supply a finite nondecreasing target inside forcing coverage",
                    self.time()
                ),
            ));
        }
        let mut count = 0;
        let status = loop {
            if self.time() == target {
                break AdvanceStatus::ReachedTarget;
            }
            if cancelled() {
                break AdvanceStatus::Cancelled;
            }
            if count == max_steps {
                break AdvanceStatus::StepLimit;
            }
            let next = self.next_accepted(target)?;
            let projected = if observer.is_some() {
                let step = next
                    .last
                    .as_ref()
                    .map(|receipt| receipt.project(&self.context, &next.land))
                    .transpose()?;
                Some((step, next.flow.to_flow(self.domain())?))
            } else {
                None
            };
            self.accepted = next;
            count += 1;
            if let (Some(observer), Some((Some(step), state))) = (&mut observer, projected) {
                observer(&step, state.cells());
            }
        };
        Ok(AdvanceReport {
            status,
            accepted_time: self.time(),
            accepted_steps: count,
        })
    }
}
