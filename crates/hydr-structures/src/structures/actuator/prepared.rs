//! The prepared old-target interval of a two-phase actuator step and its preparation.

use super::{GateActuatorBatch, GateActuatorEvent, GateActuatorOutput};
use crate::structures::numerics::actuator::{TensorActuatorLeft, left};
use crate::structures::{StructureError, StructureInputErrorKind};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Old-target response over one event-free interval, before any endpoint target packet.
///
/// `GateActuatorBatch::prepare_to` creates it. `GateActuatorBatch::complete` consumes it.
/// Dropping it leaves the accepted batch unchanged.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct GateActuatorPreparedInterval<R: Runtime<DType = DType>> {
    pub(crate) base: GateActuatorBatch<R>,
    pub(crate) left: TensorActuatorLeft<R>,
    pub(crate) frame: GateActuatorBatch<R>,
    pub(crate) next_target: usize,
}

impl<R: Runtime<DType = DType>> GateActuatorPreparedInterval<R> {
    /// Accepted start time in seconds.
    #[doc(hidden)]
    pub fn start_s(&self) -> f64 {
        self.left.start
    }
    /// Interval end in seconds.
    #[doc(hidden)]
    pub fn end_s(&self) -> f64 {
        self.left.end
    }
    /// Event exactly at the interval end. `None` when the end lies before the next event.
    #[doc(hidden)]
    pub fn event(&self) -> Option<GateActuatorEvent> {
        self.left.event
    }
    /// Accepted batch at the interval start.
    #[doc(hidden)]
    pub fn start(&self) -> &GateActuatorBatch<R> {
        &self.base
    }
    /// Project the old-target position at the interval end, before any endpoint packet.
    #[doc(hidden)]
    pub fn left_outputs(&self) -> Result<Vec<GateActuatorOutput>, StructureError> {
        self.frame.outputs()
    }
}

impl<R: Runtime<DType = DType>> GateActuatorBatch<R> {
    /// Advance the old target to `end_s` without an endpoint target packet.
    ///
    /// The interval must contain no event. An end exactly at the next event is aligned to it.
    /// The accepted batch stays unchanged.
    #[doc(hidden)]
    pub fn prepare_to(&self, end_s: f64) -> Result<GateActuatorPreparedInterval<R>, StructureError>
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
        self.prepare(end_s, false)
    }

    /// Advance the old target to `end_s` without an endpoint target packet, counting each
    /// arrival window as one event.
    ///
    /// The end must lie at or before `next_coordinated_event`. It may lie inside an arrival
    /// window ahead, and it may pass the upper bound of a window the accepted time lies inside.
    /// The prepared event holds every target, arrival bound, and terminal flag exactly at the
    /// end. The accepted batch stays unchanged.
    #[doc(hidden)]
    pub fn prepare_coordinated_to(
        &self,
        end_s: f64,
    ) -> Result<GateActuatorPreparedInterval<R>, StructureError>
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
        self.prepare(end_s, true)
    }

    /// The prepared interval to `end_s`, bounded by the standalone or the `coordinated` event.
    fn prepare(
        &self,
        end_s: f64,
        coordinated: bool,
    ) -> Result<GateActuatorPreparedInterval<R>, StructureError>
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
        let next = if coordinated {
            self.next_coordinated_event()?
        } else {
            self.next_event()?
        };
        let next = next.ok_or_else(|| {
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
        let event = if coordinated {
            self.event_at(end_s)?
        } else if end_s == next.time_s {
            Some(next)
        } else {
            None
        };
        let steps = self.steps.checked_add(1).ok_or_else(|| {
            StructureError::numerical("actuator accepted interval counter overflow")
        })?;
        let next_target = if event.and_then(|e| e.target_index).is_some() {
            self.next_target.checked_add(1).ok_or_else(|| {
                StructureError::numerical("actuator target event counter overflow")
            })?
        } else {
            self.next_target
        };
        let left = left(&self.context, &self.state, self.time, end_s, event)?;
        let frame = Self {
            context: Arc::clone(&self.context),
            state: left.state.clone(),
            time: end_s,
            next_target: self.next_target,
            steps,
            last: self.last.clone(),
        };
        Ok(GateActuatorPreparedInterval {
            base: self.clone(),
            left,
            frame,
            next_target,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::interval::tests::{accepted, target, update};
    use crate::structures::interval_fixture::{
        Cpu, TestResult, actuators, actuators_at, bits, context,
    };
    use crate::structures::{GateActuatorBatch, StructureError, StructureInputErrorKind};

    /// Target installed at 1 s. At 0.3 m/s it arrives near 1.667 s, before the target time at
    /// 2 s. The rounded rate and clock leave its window two distinct bounds.
    const ARRIVING_M: f64 = 0.2;

    /// The batch moving toward `ARRIVING_M` from 1 s, with its arrival window `[lower, upper]`.
    fn moving() -> TestResult<(GateActuatorBatch<Cpu>, f64, f64)> {
        let context = context()?;
        let batch = actuators_at(&context, "g1", (0.3, 0.3), (0.0, 0.0))?
            .trial_to(1.0, &update(1.0, ARRIVING_M))?
            .into_next();
        let output = batch.outputs()?.remove(0);
        let lower = output.arrival_lower_s.ok_or("a lower bound")?;
        let upper = output.arrival_upper_s.ok_or("an upper bound")?;
        Ok((batch, lower, upper))
    }

    fn rejects_unsupported<T>(result: Result<T, StructureError>) -> TestResult {
        let error = result.err().ok_or("the interval rejects")?;
        assert_eq!(
            error.input_kind(),
            Some(StructureInputErrorKind::UnsupportedRegime)
        );
        Ok(())
    }

    /// Frozen criteria, fixed before the first run:
    /// - The window holds two distinct bounds `1 < l < u < 2`.
    /// - The standalone walk reports `l`. The coordinated walk reports `u` with only the upper
    ///   flag set.
    /// - One coordinated interval to `u` accepts with that event. It ends at the target bit for
    ///   bit, and the arrival count becomes 1.
    /// - The coordinated walk then reports the target time 2 s.
    /// - The standalone `prepare_to(u)` rejects with `UnsupportedRegime`.
    #[test]
    fn a_coordinated_interval_reaches_an_arrival_window_in_one_end() -> TestResult {
        let (batch, lower, upper) = moving()?;
        assert!(
            1.0 < lower && lower < upper && upper < 2.0,
            "[{lower},{upper}]"
        );
        let standalone = batch.next_event()?.ok_or("a standalone event")?;
        assert_eq!(standalone.time_s, lower);
        let event = batch
            .next_coordinated_event()?
            .ok_or("a coordinated event")?;
        assert_eq!(event.time_s, upper);
        assert!(event.arrival_upper_boundary && !event.arrival_lower_boundary);
        assert_eq!(event.target_index, None);
        let prepared = batch.prepare_coordinated_to(upper)?;
        assert_eq!(prepared.event(), Some(event));
        let next = batch.complete(prepared, &[])?.into_next();
        let output = next.outputs()?.remove(0);
        assert_eq!(output.position.opening_m.to_bits(), ARRIVING_M.to_bits());
        assert_eq!(next.states()?[0].arrival_count, 1);
        let after = next.next_coordinated_event()?.ok_or("the target time")?;
        assert_eq!((after.time_s, after.target_index), (2.0, Some(1)));
        rejects_unsupported(batch.prepare_to(upper))
    }

    /// Frozen criteria, fixed before the first run:
    /// - The window holds a time `m` strictly between its bounds. A coordinated interval to `m`
    ///   accepts with no event.
    /// - From `m`, the coordinated walk reports the target time 2 s.
    /// - A coordinated interval from `m` to 1.9 s accepts with no event and ends at the target
    ///   bit for bit. The standalone `prepare_to(1.9)` from `m` rejects with `UnsupportedRegime`.
    /// - From 1 s, `prepare_coordinated_to(1.9)` rejects with `UnsupportedRegime`: that interval
    ///   holds the whole window.
    #[test]
    fn an_interval_from_inside_a_window_crosses_its_upper_bound() -> TestResult {
        let (batch, lower, upper) = moving()?;
        let inside = f64::from_bits(lower.to_bits() + 1);
        assert!(lower < inside && inside < upper, "[{lower},{upper}]");
        let prepared = batch.prepare_coordinated_to(inside)?;
        assert_eq!(prepared.event(), None);
        let within = batch.complete(prepared, &[])?.into_next();
        let event = within.next_coordinated_event()?.ok_or("the target time")?;
        assert_eq!((event.time_s, event.target_index), (2.0, Some(1)));
        let prepared = within.prepare_coordinated_to(1.9)?;
        assert_eq!(prepared.event(), None);
        let past = within.complete(prepared, &[])?.into_next();
        let output = past.outputs()?.remove(0);
        assert_eq!(output.position.opening_m.to_bits(), ARRIVING_M.to_bits());
        rejects_unsupported(within.prepare_to(1.9))?;
        rejects_unsupported(batch.prepare_coordinated_to(1.9))
    }

    /// Frozen criterion, fixed before the first run: at every end of the standalone walk of the
    /// target fixture, the coordinated interval holds the same event, states, outputs, and
    /// diagnostics bit for bit.
    #[test]
    fn a_coordinated_interval_at_a_standalone_end_keeps_every_bit() -> TestResult {
        let context = context()?;
        let mut batch = actuators(&context, "g1", 0.0, 0.0)?;
        while let Some(event) = batch.next_event()? {
            let packets = match event.target_index {
                Some(_) => update(event.time_s, target(event.time_s)),
                None => Vec::new(),
            };
            let coordinated = batch.prepare_coordinated_to(event.time_s)?;
            assert_eq!(coordinated.event(), Some(event));
            let one = batch.complete(batch.prepare_to(event.time_s)?, &packets)?;
            let two = batch.complete(coordinated, &packets)?;
            assert_eq!(bits(&one.states()?), bits(&two.states()?));
            assert_eq!(bits(&one.outputs()?), bits(&two.outputs()?));
            assert_eq!(bits(&one.diagnostics()?), bits(&two.diagnostics()?));
            assert_eq!(accepted(&one.next)?, accepted(&two.next)?);
            batch = one.into_next();
        }
        assert_eq!(batch.time(), 4.0);
        Ok(())
    }
}
