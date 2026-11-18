//! Two-phase controlled gate intervals: old-target response first, decision samples second.

use super::{ControlledGateBatch, ControlledGateEvent, ControlledGateTrial};
use crate::structures::control::OperatingControlPreparedInterval;
use crate::structures::{
    ControlSample, GateActuatorPreparedInterval, StructureError, StructureInputErrorKind,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Old-target gate response over one event-free interval, before its decision samples.
///
/// `ControlledGateBatch::prepare_to` creates it. `ControlledGateBatch::complete` consumes it.
/// Dropping it leaves the accepted batch unchanged.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ControlledGatePreparedInterval<R: Runtime<DType = DType>> {
    pub(crate) base: ControlledGateBatch<R>,
    pub(crate) controls: OperatingControlPreparedInterval<R>,
    pub(crate) actuators: GateActuatorPreparedInterval<R>,
    pub(crate) event: Option<ControlledGateEvent>,
}

impl<R: Runtime<DType = DType>> ControlledGatePreparedInterval<R> {
    /// Accepted start time in seconds.
    #[doc(hidden)]
    pub fn start_s(&self) -> f64 {
        self.base.time()
    }
    /// Interval end in seconds.
    #[doc(hidden)]
    pub fn end_s(&self) -> f64 {
        self.controls.end_s
    }
    /// Common event exactly at the interval end. `None` when the end lies before the next event.
    #[doc(hidden)]
    pub fn event(&self) -> Option<ControlledGateEvent> {
        self.event
    }
    /// Accepted batch at the interval start.
    #[doc(hidden)]
    pub fn start(&self) -> &ControlledGateBatch<R> {
        &self.base
    }
    /// Prepared old-target actuator response, the source of the interval gate operation.
    #[doc(hidden)]
    pub fn actuators(&self) -> &GateActuatorPreparedInterval<R> {
        &self.actuators
    }
}

impl<R: Runtime<DType = DType>> ControlledGateBatch<R> {
    /// Advance the actual gate response to `end_s` under its old targets, without decision samples.
    ///
    /// The interval must contain no decision, arrival boundary, or terminal event.
    /// An end exactly at the next event is aligned to it. The accepted batch stays unchanged.
    #[doc(hidden)]
    pub fn prepare_to(
        &self,
        end_s: f64,
    ) -> Result<ControlledGatePreparedInterval<R>, StructureError>
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
        let duration = end_s - self.time();
        if !end_s.is_finite() || !duration.is_finite() || duration <= 0.0 {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!(
                    "controlled gate interval=[{},{}],duration={duration}: supply a finite positive original duration",
                    self.time(),
                    end_s
                ),
            ));
        }
        let event = self.next_event()?.ok_or_else(|| {
            StructureError::input(
                StructureInputErrorKind::UnsupportedRegime,
                "controlled gate trial exceeds terminal support: retain the original common horizon",
            )
        })?;
        if end_s > event.time_s {
            return Err(StructureError::input(
                StructureInputErrorKind::UnsupportedRegime,
                "controlled gate interval crosses its next event: split at the common event",
            ));
        }
        let controls = self.controls.prepare_to(end_s)?;
        let actuators = self.actuators.prepare_to(end_s)?;
        Ok(ControlledGatePreparedInterval {
            base: self.clone(),
            controls,
            actuators,
            event: (end_s == event.time_s).then_some(event),
        })
    }

    /// Apply the decision samples of a prepared interval, then install each new target on its right side.
    ///
    /// The prepared interval must come from this accepted batch.
    /// An end without a decision takes an empty sample slice.
    #[doc(hidden)]
    pub fn complete(
        &self,
        prepared: ControlledGatePreparedInterval<R>,
        decision_samples: &[ControlSample],
    ) -> Result<ControlledGateTrial<R>, StructureError>
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
        if !Arc::ptr_eq(&self.context, &prepared.base.context) {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                "prepared controlled gate interval belongs to another binding: prepare it from this batch",
            ));
        }
        let end_s = prepared.controls.end_s;
        let decision = prepared
            .event
            .is_some_and(|event| event.decision_index.is_some());
        let controls = self
            .controls
            .complete(prepared.controls, decision_samples)?
            .into_next();
        let update = if decision {
            Some(self.context.update(&controls, &self.actuators, end_s)?)
        } else {
            None
        };
        let actuators = self
            .actuators
            .complete_with_update(prepared.actuators, update)?
            .into_next();
        Ok(ControlledGateTrial {
            next: Self {
                context: Arc::clone(&self.context),
                controls,
                actuators,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::structures::interval_fixture::{
        Cpu, TestResult, bits, context, controlled_gate, samples,
    };
    use crate::structures::{ControlledGateBatch, StructureInputErrorKind};

    fn accepted(batch: &ControlledGateBatch<Cpu>) -> TestResult<String> {
        Ok(bits(&(
            batch.time().to_bits(),
            batch.accepted_steps(),
            batch.controls().states()?,
            batch.controls().last_step()?,
            batch.actuators().states()?,
            batch.actuators().outputs()?,
            batch.actuators().last_step()?,
        )))
    }

    /// Stage above the upper threshold opens the gate at 1 s and 2 s. Stage below the lower closes it at 3 s.
    fn stage(time_s: f64) -> f64 {
        if time_s < 2.5 { 3.0 } else { 0.5 }
    }

    #[test]
    fn prepare_then_complete_equals_one_phase_trial_bit_for_bit() -> TestResult {
        let context = context()?;
        let mut batch = controlled_gate(&context)?;
        let mut decisions = 0;
        while let Some(event) = batch.next_event()? {
            let midpoint = batch.time() + (event.time_s - batch.time()) * 0.5;
            for end_s in [midpoint, event.time_s] {
                let decision = batch
                    .prepare_to(end_s)?
                    .event()
                    .is_some_and(|event| event.decision_index.is_some());
                let input = if decision {
                    decisions += 1;
                    samples("g1", end_s, stage(end_s))
                } else {
                    Vec::new()
                };
                let one = batch.trial_to(end_s, &input)?;
                let two = batch.complete(batch.prepare_to(end_s)?, &input)?;
                assert_eq!(accepted(&one.next)?, accepted(&two.next)?);
                batch = one.into_next();
            }
        }
        assert_eq!(decisions, 3);
        assert_eq!(batch.time(), 4.0);
        Ok(())
    }

    #[test]
    fn the_decision_command_acts_only_on_the_right_side() -> TestResult {
        let context = context()?;
        let batch = controlled_gate(&context)?;
        let prepared = batch.prepare_to(1.0)?;
        let event = prepared.event().ok_or("decision aligns at 1 s")?;
        assert_eq!(event.decision_index, Some(0));
        let left = prepared.actuators().left_outputs()?.remove(0);
        assert_eq!(left.target_opening_m, 0.0);
        assert_eq!(left.position.opening_m, 0.0);
        let trial = batch.complete(prepared, &samples("g1", 1.0, 3.0))?;
        let receipt = trial
            .actuators()
            .last_step()?
            .ok_or("trial keeps its actuator step")?
            .receipts
            .remove(0);
        assert_eq!(bits(&receipt.left_position), bits(&left.position));
        assert_eq!(trial.actuators().outputs()?[0].target_opening_m, 1.0);
        Ok(())
    }

    #[test]
    fn dropping_or_misusing_a_prepared_interval_leaves_the_batch_unchanged() -> TestResult {
        let context = context()?;
        let batch = controlled_gate(&context)?
            .trial_to(1.0, &samples("g1", 1.0, 3.0))?
            .into_next();
        let before = accepted(&batch)?;
        drop(batch.prepare_to(2.0)?);
        assert_eq!(accepted(&batch)?, before);
        let error = batch
            .complete(batch.prepare_to(2.0)?, &[])
            .err()
            .ok_or("missing decision samples reject")?;
        assert!(error.input_kind().is_some());
        assert_eq!(accepted(&batch)?, before);
        let other = controlled_gate(&context)?;
        let error = other
            .complete(controlled_gate(&context)?.prepare_to(0.5)?, &[])
            .err()
            .ok_or("another binding rejects")?;
        assert_eq!(error.input_kind(), Some(StructureInputErrorKind::Identity));
        let crossing = batch.prepare_to(2.5).err().ok_or("crossing rejects")?;
        assert_eq!(
            crossing.input_kind(),
            Some(StructureInputErrorKind::UnsupportedRegime)
        );
        Ok(())
    }
}
