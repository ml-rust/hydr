//! Two-phase actuator intervals: the old-target left state first, the endpoint packet second.
//!
//! The sibling `prepared` module prepares the left state. This module installs the packet.

use super::{GateActuatorBatch, GateActuatorPreparedInterval, GateActuatorTrial, GateTargetUpdate};
use crate::structures::numerics::actuator::install;
use crate::structures::tensor::actuator::{
    TensorActuatorContext, TensorActuatorUpdate, record::TensorRecord,
};
use crate::structures::{StructureError, StructureInputErrorKind};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> GateActuatorBatch<R> {
    /// Install the endpoint target packets on the right side of a prepared interval.
    ///
    /// The prepared interval must come from this accepted batch.
    /// An end without a target event takes an empty packet slice.
    #[doc(hidden)]
    pub fn complete(
        &self,
        prepared: GateActuatorPreparedInterval<R>,
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
        let end_s = prepared.left.end;
        self.complete_admitting(prepared, |context, expected| {
            crate::structures::numerics::actuator::admit_update(
                context,
                target_updates,
                end_s,
                expected,
            )
        })
    }

    /// Install a complete tensor update admitted by the bound command owner.
    pub(crate) fn complete_with_update(
        &self,
        prepared: GateActuatorPreparedInterval<R>,
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
        self.complete_admitting(prepared, |context, expected| match (expected, update) {
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

    fn complete_admitting(
        &self,
        prepared: GateActuatorPreparedInterval<R>,
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
        if !self.same_accepted(&prepared.base) {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                "prepared actuator interval belongs to another accepted state: prepare it from this batch",
            ));
        }
        let expected = prepared
            .left
            .event
            .and_then(|event| event.target_index)
            .is_some();
        let update = admit(&self.context, expected)?;
        let steps = prepared.frame.steps;
        let step = install(&self.context, &self.state, prepared.left, update)?;
        let next = Self {
            context: Arc::clone(&self.context),
            state: step.after.clone(),
            time: step.end,
            next_target: prepared.next_target,
            steps,
            last: Some(Arc::new(step)),
        };
        Ok(GateActuatorTrial { next })
    }

    /// Same context, clock, counters, and position storage as `other`.
    pub(crate) fn same_accepted(&self, other: &Self) -> bool {
        let (a, b) = (&self.state, &other.state);
        Arc::ptr_eq(&self.context, &other.context)
            && self.time.to_bits() == other.time.to_bits()
            && self.steps == other.steps
            && self.next_target == other.next_target
            && a.position.center.shares_storage_with(&b.position.center)
            && a.position.lower.shares_storage_with(&b.position.lower)
            && a.position.upper.shares_storage_with(&b.position.upper)
            && a.position.error.shares_storage_with(&b.position.error)
            && a.target.shares_storage_with(&b.target)
            && a.anchor.epoch.shares_storage_with(&b.anchor.epoch)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use crate::structures::interval_fixture::{Cpu, TestResult, actuators, bits, context};
    use crate::structures::{GateActuatorBatch, GateTargetUpdate, StructureInputErrorKind};

    /// The update packet of actuator `g1` toward `target_opening_m`, issued at `time_s`.
    pub(in super::super) fn update(time_s: f64, target_opening_m: f64) -> Vec<GateTargetUpdate> {
        vec![GateTargetUpdate {
            actuator_id: "actuator-g1".into(),
            structure_id: "g1".into(),
            target_opening_m,
            issued_at_s: time_s,
            available_at_s: time_s,
        }]
    }

    /// Debug text of every accepted field of `batch`.
    pub(in super::super) fn accepted(batch: &GateActuatorBatch<Cpu>) -> TestResult<String> {
        Ok(bits(&(
            batch.time().to_bits(),
            batch.accepted_steps(),
            batch.states()?,
            batch.outputs()?,
            batch.last_step()?,
        )))
    }

    /// Target for each original event: open, repeat, then close to the terminal bound.
    pub(in super::super) fn target(time_s: f64) -> f64 {
        if time_s < 2.5 { 0.5 } else { 0.0 }
    }

    #[test]
    fn prepare_then_complete_equals_one_phase_trial_bit_for_bit() -> TestResult {
        let context = context()?;
        let mut batch = actuators(&context, "g1", 0.0, 0.0)?;
        let mut intervals = 0;
        while let Some(event) = batch.next_event()? {
            let midpoint = batch.time() + (event.time_s - batch.time()) * 0.5;
            for end_s in [midpoint, event.time_s] {
                let packets = match batch.prepare_to(end_s)?.event() {
                    Some(event) if event.target_index.is_some() => update(end_s, target(end_s)),
                    _ => Vec::new(),
                };
                let one = batch.trial_to(end_s, &packets)?;
                let two = batch.complete(batch.prepare_to(end_s)?, &packets)?;
                assert_eq!(bits(&one.states()?), bits(&two.states()?));
                assert_eq!(bits(&one.outputs()?), bits(&two.outputs()?));
                assert_eq!(bits(&one.diagnostics()?), bits(&two.diagnostics()?));
                assert_eq!(accepted(&one.next)?, accepted(&two.next)?);
                batch = one.into_next();
                intervals += 1;
            }
        }
        assert_eq!(batch.time(), 4.0);
        assert!(intervals >= 8);
        Ok(())
    }

    #[test]
    fn dropping_a_prepared_interval_restores_the_accepted_batch_exactly() -> TestResult {
        let context = context()?;
        let batch = actuators(&context, "g1", 0.0, 0.0)?
            .trial_to(1.0, &update(1.0, 0.5))?
            .into_next();
        let before = accepted(&batch)?;
        let untouched = batch.clone();
        for end_s in [1.5, 2.0] {
            let prepared = batch.prepare_to(end_s)?;
            assert!(!prepared.left_outputs()?.is_empty());
            drop(prepared);
            assert_eq!(accepted(&batch)?, before);
        }
        let after = batch.trial_to(2.0, &update(2.0, 0.5))?;
        let reference = untouched.trial_to(2.0, &update(2.0, 0.5))?;
        assert_eq!(bits(&after.states()?), bits(&reference.states()?));
        assert_eq!(bits(&after.diagnostics()?), bits(&reference.diagnostics()?));
        Ok(())
    }

    #[test]
    fn events_at_the_end_align_and_an_interval_containing_one_rejects() -> TestResult {
        let context = context()?;
        let batch = actuators(&context, "g1", 0.0, 0.0)?;
        let event = batch
            .next_event()?
            .ok_or("fresh actuator has a next event")?;
        assert_eq!(event.time_s, 1.0);
        assert_eq!(batch.prepare_to(1.0)?.event(), Some(event));
        assert_eq!(batch.prepare_to(0.5)?.event(), None);
        let crossing = batch
            .prepare_to(1.5)
            .err()
            .ok_or("crossing interval rejects")?;
        assert_eq!(
            crossing.input_kind(),
            Some(StructureInputErrorKind::UnsupportedRegime)
        );
        let empty = batch
            .prepare_to(0.0)
            .err()
            .ok_or("empty interval rejects")?;
        assert_eq!(
            empty.input_kind(),
            Some(StructureInputErrorKind::Definition)
        );
        let missing = batch
            .complete(batch.prepare_to(1.0)?, &[])
            .err()
            .ok_or("target event without packets rejects")?;
        assert_eq!(
            missing.input_kind(),
            Some(StructureInputErrorKind::OperatingPoint)
        );
        Ok(())
    }

    #[test]
    fn complete_rejects_a_prepared_left_state_from_another_accepted_batch() -> TestResult {
        let context = context()?;
        let fresh = actuators(&context, "g1", 0.0, 0.0)?;
        let opened = fresh.trial_to(1.0, &update(1.0, 0.5))?.into_next();
        let closed = fresh.trial_to(1.0, &update(1.0, 0.0))?.into_next();
        let prepared = opened.prepare_to(1.5)?;
        let error = closed
            .complete(prepared.clone(), &[])
            .err()
            .ok_or("another left state rejects")?;
        assert_eq!(error.input_kind(), Some(StructureInputErrorKind::Identity));
        assert!(opened.complete(prepared, &[]).is_ok());
        let rebuilt = actuators(&context, "g1", 0.0, 0.0)?;
        let error = rebuilt
            .complete(fresh.prepare_to(0.5)?, &[])
            .err()
            .ok_or("another context rejects")?;
        assert_eq!(error.input_kind(), Some(StructureInputErrorKind::Identity));
        Ok(())
    }

    #[test]
    fn the_left_state_keeps_the_old_target_and_the_packet_acts_on_the_right() -> TestResult {
        let context = context()?;
        let batch = actuators(&context, "g1", 0.0, 0.0)?
            .trial_to(1.0, &update(1.0, 0.5))?
            .into_next()
            .trial_to(2.0, &update(2.0, 0.5))?
            .into_next();
        let prepared = batch.prepare_to(3.0)?;
        let left = prepared.left_outputs()?.remove(0);
        assert_eq!(left.target_opening_m, 0.5);
        assert_eq!(left.time_s, 3.0);
        let trial = batch.complete(prepared, &update(3.0, 0.0))?;
        let receipt = trial.diagnostics()?.receipts.remove(0);
        assert_eq!(bits(&receipt.left_position), bits(&left.position));
        assert!(receipt.target_changed);
        assert_eq!(trial.outputs()?[0].target_opening_m, 0.0);
        Ok(())
    }
}
