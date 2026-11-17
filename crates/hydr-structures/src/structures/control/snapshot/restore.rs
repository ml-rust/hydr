//! Atomic complete snapshot checks without full historical decision replay.

use super::super::OperatingControlBatch;
use super::compatibility::require;
use super::{
    OPERATING_CONTROL_CONTRACT_VERSION, OPERATING_CONTROL_SNAPSHOT_SCHEMA,
    OperatingControlSnapshot, OperatingControlSnapshotData,
};
use crate::structures::{StructureError, StructureInputErrorKind};
use hydr_core::contract::package_identity;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> OperatingControlBatch<R> {
    /// Restore trusted tensor continuation while retaining this batch's complete original bound context.
    pub fn restore(
        &mut self,
        snapshot: &OperatingControlSnapshot<R>,
    ) -> Result<(), StructureError> {
        super::compatibility::check_batch(self, &snapshot.batch)?;
        self.state = snapshot.batch.state.clone();
        self.time = snapshot.batch.time;
        self.next_decision = snapshot.batch.next_decision;
        self.steps = snapshot.batch.steps;
        self.last = snapshot.batch.last.clone();
        Ok(())
    }
    /// Check complete caller-owned continuation before atomically replacing accepted values.
    pub fn restore_data(
        &mut self,
        data: &OperatingControlSnapshotData,
    ) -> Result<(), StructureError>
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
        package_identity(
            "operating control",
            &data.schema,
            OPERATING_CONTROL_SNAPSHOT_SCHEMA,
            &data.contract,
            OPERATING_CONTROL_CONTRACT_VERSION,
        )
        .map_err(StructureError::definition)?;
        super::compatibility::check(self, data)?;
        require(
            data.time_s.is_finite()
                && data.time_s >= self.schedule().initial_time_s
                && data.time_s <= self.schedule().end_s,
            "imported control time is outside original support",
        )?;
        let processed = self
            .schedule()
            .decision_times_s
            .partition_point(|time| *time <= data.time_s);
        require(
            data.next_decision_index == processed && data.accepted_steps >= processed,
            "imported decision index or accepted interval count disagrees with original schedule",
        )?;
        require(
            (data.accepted_steps == 0) == data.last_step.is_none(),
            "imported interval count disagrees with retained last step",
        )?;
        if data.accepted_steps == 0 {
            require(
                data.time_s.to_bits() == self.schedule().initial_time_s.to_bits(),
                "zero-step control state changes its original initial time",
            )?;
        }
        let state = super::state_check::check(self, &data.states, data.time_s, processed)?;
        let mut candidate = Self {
            context: Arc::clone(&self.context),
            state,
            time: data.time_s,
            next_decision: processed,
            steps: data.accepted_steps,
            last: None,
        };
        if let Some(last) = &data.last_step {
            require(
                last.start_s.is_finite()
                    && last.start_s >= self.schedule().initial_time_s
                    && last.end_s.to_bits() == data.time_s.to_bits()
                    && last.start_s < last.end_s,
                "retained control step does not end at accepted time with a valid original start",
            )?;
            require(
                if data.accepted_steps == 1 {
                    last.start_s.to_bits() == self.schedule().initial_time_s.to_bits()
                } else {
                    last.start_s > self.schedule().initial_time_s
                },
                "retained control step start contradicts accepted interval count",
            )?;
            let prior_processed = self
                .schedule()
                .decision_times_s
                .partition_point(|time| *time <= last.start_s);
            require(
                data.accepted_steps
                    .checked_sub(1)
                    .is_some_and(|prior_steps| prior_steps >= prior_processed),
                "retained prior interval count cannot cover its processed decisions",
            )?;
            let before = if last.decision_index.is_some() {
                require(
                    last.decisions.len() == self.definitions().len(),
                    "retained control decision receipts are incomplete",
                )?;
                let values = last
                    .decisions
                    .iter()
                    .map(|receipt| receipt.before.clone())
                    .collect::<Vec<_>>();
                super::state_check::check(self, &values, last.start_s, prior_processed)?
            } else {
                require(
                    last.decisions.is_empty(),
                    "non-decision control step contains decision receipts",
                )?;
                super::state_check::check(self, &data.states, last.start_s, prior_processed)?
            };
            let before = Self {
                context: Arc::clone(&self.context),
                state: before,
                time: last.start_s,
                next_decision: prior_processed,
                steps: data.accepted_steps - 1,
                last: None,
            };
            let samples = last
                .decisions
                .iter()
                .map(|receipt| receipt.sample.clone())
                .collect::<Vec<_>>();
            let trial = before
                .trial_to(last.end_s, &samples)
                .map_err(super::state_check::as_definition)?;
            require(
                super::receipt_bits::step(&trial.diagnostics()?, last),
                "retained last control diagnostics differ from recomputed immediate trial",
            )?;
            require(
                super::receipt_bits::states(&trial.states()?, &data.states),
                "imported accepted control state differs from retained last trial",
            )?;
            candidate.last = trial.into_next().last;
        } else {
            let expected = crate::structures::tensor::control::TensorControlState::initial(
                &candidate.context,
            )?;
            require(
                super::receipt_bits::states(
                    &super::super::projection::states(&candidate.context, &expected)?,
                    &data.states,
                ),
                "zero-step control values differ from original assumed initial state",
            )?;
        }
        if candidate.next_decision != processed {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                "imported control decision continuation differs after receipt checks",
            ));
        }
        *self = candidate;
        Ok(())
    }
}
