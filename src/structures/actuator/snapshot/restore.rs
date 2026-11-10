//! Atomic complete continuation restoration with one bounded immediate-step replay.

use super::super::*;
use super::compatibility::{as_definition, require};
use super::receipt_bits::ReceiptBits;
use crate::structures::StructureError;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> GateActuatorBatch<R> {
    /// Restore trusted tensors while retaining the complete destination context.
    pub fn restore(&mut self, snapshot: &GateActuatorSnapshot<R>) -> Result<(), StructureError> {
        super::compatibility::trusted(self, &snapshot.batch)?;
        self.state = snapshot.batch.state.clone();
        self.time = snapshot.batch.time;
        self.next_target = snapshot.batch.next_target;
        self.steps = snapshot.batch.steps;
        self.last = snapshot.batch.last.clone();
        Ok(())
    }
    /// Check complete caller-owned continuation before atomically replacing accepted values.
    pub fn restore_data(&mut self, data: &GateActuatorSnapshotData) -> Result<(), StructureError>
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
        self.restore_checked(data).map_err(as_definition)
    }
    fn restore_checked(&mut self, data: &GateActuatorSnapshotData) -> Result<(), StructureError>
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
        super::compatibility::check(self, data)?;
        require(
            data.time_s.is_finite()
                && data.time_s >= self.schedule().initial_time_s
                && data.time_s <= self.schedule().end_s,
            "imported actuator time is outside original finite support",
        )?;
        let processed = self
            .schedule()
            .target_times_s
            .partition_point(|t| *t <= data.time_s);
        require(
            data.next_target_index == processed
                && data.accepted_steps >= processed
                && (data.accepted_steps == 0) == data.last_step.is_none(),
            "imported actuator event index, interval count, or last-step presence is inconsistent",
        )?;
        if data.accepted_steps == 0 {
            require(
                data.time_s.same_bits(&self.schedule().initial_time_s),
                "zero-step actuator continuation changed its original initial time",
            )?;
        }
        let state = super::state_check::check(
            self,
            &data.states,
            data.time_s,
            processed,
            data.accepted_steps,
        )?;
        let mut candidate = Self {
            context: Arc::clone(&self.context),
            state,
            time: data.time_s,
            next_target: processed,
            steps: data.accepted_steps,
            last: None,
        };
        if let Some(last) = &data.last_step {
            require(
                last.start_s.is_finite()
                    && last.start_s >= self.schedule().initial_time_s
                    && last.start_s < last.end_s
                    && last.end_s.same_bits(&data.time_s),
                "retained actuator immediate step does not end at accepted time with a valid original start",
            )?;
            require(
                if data.accepted_steps == 1 {
                    last.start_s.same_bits(&self.schedule().initial_time_s)
                } else {
                    last.start_s > self.schedule().initial_time_s
                },
                "retained actuator step start contradicts accepted interval count",
            )?;
            let prior_processed = self
                .schedule()
                .target_times_s
                .partition_point(|t| *t <= last.start_s);
            let prior_steps = data.accepted_steps.checked_sub(1).ok_or_else(|| {
                StructureError::numerical("retained actuator step has no prior interval count")
            })?;
            require(
                prior_steps >= prior_processed && last.receipts.len() == self.definitions().len(),
                "retained actuator prior interval count or receipt vector is incomplete",
            )?;
            let rows = last
                .receipts
                .iter()
                .map(|r| r.before.clone())
                .collect::<Vec<_>>();
            let before_state =
                super::state_check::check(self, &rows, last.start_s, prior_processed, prior_steps)?;
            let before = Self {
                context: Arc::clone(&self.context),
                state: before_state,
                time: last.start_s,
                next_target: prior_processed,
                steps: prior_steps,
                last: None,
            };
            let updates = last
                .receipts
                .iter()
                .filter_map(|r| r.update.clone())
                .collect::<Vec<_>>();
            let trial = before
                .trial_to(last.end_s, &updates)
                .map_err(as_definition)?;
            require(
                trial.diagnostics()?.same_bits(last),
                "retained actuator immediate diagnostics differ from recomputed original trial",
            )?;
            require(
                trial.states()?.same_bits(&data.states),
                "imported actuator state differs from recomputed immediate trial",
            )?;
            candidate.last = trial.into_next().last;
        } else {
            let initial = crate::numerics::structures::actuator::initial(&candidate.context)
                .map_err(as_definition)?;
            require(
                super::super::projection::states(&candidate.context, &initial)?
                    .same_bits(&data.states),
                "zero-step actuator continuation differs from canonical original initial state",
            )?;
        }
        *self = candidate;
        Ok(())
    }
}
