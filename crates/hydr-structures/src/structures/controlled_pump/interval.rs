//! Two-phase pump intervals: the held operation first, the decision samples second.

use super::{ControlledPumpBatch, ControlledPumpTrial};
use crate::structures::control::OperatingControlPreparedInterval;
use crate::structures::{ControlEvent, ControlSample, StructureError, StructureOperation};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Held right-side pump operation over one event-free interval, before its decision samples.
///
/// `ControlledPumpBatch::prepare_to` creates it. `ControlledPumpBatch::complete` consumes it.
/// Dropping it leaves the accepted batch unchanged.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ControlledPumpPreparedInterval<R: Runtime<DType = DType>> {
    pub(crate) base: ControlledPumpBatch<R>,
    pub(crate) controls: OperatingControlPreparedInterval<R>,
    pub(crate) frame: ControlledPumpBatch<R>,
}

impl<R: Runtime<DType = DType>> ControlledPumpPreparedInterval<R> {
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
    /// Decision or terminal event exactly at the interval end.
    #[doc(hidden)]
    pub fn event(&self) -> Option<ControlEvent> {
        self.controls.event
    }
    /// Accepted batch at the interval start.
    #[doc(hidden)]
    pub fn start(&self) -> &ControlledPumpBatch<R> {
        &self.base
    }
    /// Project the operation held over the interval: the accepted right-side operation at its start.
    #[doc(hidden)]
    pub fn operations(&self) -> Result<Vec<StructureOperation>, StructureError>
    where
        R::Client: numr::ops::ConditionalOps<R>,
    {
        self.base.operations()
    }
}

impl<R: Runtime<DType = DType>> ControlledPumpBatch<R> {
    /// Hold the accepted operation to `end_s` without decision samples.
    ///
    /// The interval must contain no decision. An end exactly at the next decision is aligned to it.
    /// The accepted batch stays unchanged.
    #[doc(hidden)]
    pub fn prepare_to(
        &self,
        end_s: f64,
    ) -> Result<ControlledPumpPreparedInterval<R>, StructureError> {
        let controls = self.controls.prepare_to(end_s)?;
        let frame = Self {
            context: Arc::clone(&self.context),
            controls: controls.left(),
        };
        Ok(ControlledPumpPreparedInterval {
            base: self.clone(),
            controls,
            frame,
        })
    }

    /// Apply the decision samples of a prepared interval. The new command governs its right side.
    ///
    /// The prepared interval must come from this accepted batch.
    /// An end without a decision takes an empty sample slice.
    #[doc(hidden)]
    pub fn complete(
        &self,
        prepared: ControlledPumpPreparedInterval<R>,
        samples: &[ControlSample],
    ) -> Result<ControlledPumpTrial<R>, StructureError>
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
                crate::structures::StructureInputErrorKind::Identity,
                "prepared pump interval belongs to another response context: prepare it from this batch",
            ));
        }
        let controls = self
            .controls
            .complete(prepared.controls, samples)?
            .into_next();
        Ok(ControlledPumpTrial {
            next: Self {
                context: Arc::clone(&self.context),
                controls,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::structures::interval_fixture::{
        Cpu, TestResult, bits, context, controlled_pump, samples,
    };
    use crate::structures::{
        ControlledPumpBatch, StructureInputErrorKind, StructureOperatingPoint,
    };

    fn accepted(batch: &ControlledPumpBatch<Cpu>) -> TestResult<String> {
        Ok(bits(&(
            batch.time().to_bits(),
            batch.accepted_steps(),
            batch.controls().states()?,
            batch.controls().last_step()?,
            batch.outputs()?,
        )))
    }

    #[test]
    fn prepare_then_complete_equals_one_phase_trial_bit_for_bit() -> TestResult {
        let context = context()?;
        let mut batch = controlled_pump(&context)?;
        while let Some(event) = batch.next_event() {
            let midpoint = batch.time() + (event.time_s - batch.time()) * 0.5;
            for end_s in [midpoint, event.time_s] {
                let input = match batch.prepare_to(end_s)?.event() {
                    Some(event) if event.decision_index.is_some() => {
                        samples("p2", end_s, if end_s < 2.5 { 3.0 } else { 0.5 })
                    }
                    _ => Vec::new(),
                };
                let one = batch.trial_to(end_s, &input)?;
                let two = batch.complete(batch.prepare_to(end_s)?, &input)?;
                assert_eq!(accepted(&one.next)?, accepted(&two.next)?);
                batch = one.into_next();
            }
        }
        assert_eq!(batch.time(), 4.0);
        Ok(())
    }

    #[test]
    fn the_operation_is_held_over_the_interval_and_switches_on_the_right() -> TestResult {
        let context = context()?;
        let batch = controlled_pump(&context)?;
        let before = accepted(&batch)?;
        let prepared = batch.prepare_to(1.0)?;
        assert_eq!(
            prepared.event().and_then(|event| event.decision_index),
            Some(0)
        );
        let held = prepared.operations()?;
        assert_eq!(
            held[0].operating_point,
            StructureOperatingPoint::PumpStoppedIsolated
        );
        let trial = batch.complete(prepared.clone(), &samples("p2", 1.0, 3.0))?;
        assert_eq!(
            trial.operations()?[0].operating_point,
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 }
        );
        assert_eq!(bits(&prepared.operations()?), bits(&held));
        drop(prepared);
        assert_eq!(accepted(&batch)?, before);
        Ok(())
    }

    #[test]
    fn complete_rejects_another_accepted_state_and_a_crossing_interval() -> TestResult {
        let context = context()?;
        let fresh = controlled_pump(&context)?;
        let running = fresh.trial_to(1.0, &samples("p2", 1.0, 3.0))?.into_next();
        let stopped = fresh.trial_to(1.0, &samples("p2", 1.0, 0.5))?.into_next();
        let error = stopped
            .complete(running.prepare_to(1.5)?, &[])
            .err()
            .ok_or("another accepted latch rejects")?;
        assert_eq!(error.input_kind(), Some(StructureInputErrorKind::Identity));
        let crossing = fresh.prepare_to(1.5).err().ok_or("crossing rejects")?;
        assert_eq!(
            crossing.input_kind(),
            Some(StructureInputErrorKind::UnsupportedRegime)
        );
        Ok(())
    }
}
