//! Caller-owned timing witness upload after finite-value checks.

use super::{TensorControlContext, TensorControlTimer};
use crate::structures::{ControlTimerWitness, StructureError, StructureInputErrorKind};
use numr::dtype::DType;
use numr::runtime::Runtime;

pub(crate) fn timers<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    timers: &[Option<ControlTimerWitness>],
) -> Result<TensorControlTimer<R>, StructureError> {
    for (lane, timer) in timers.iter().enumerate() {
        if let Some(timer) = timer {
            let finite = [
                timer.started_at_s,
                timer.evaluated_at_s,
                timer.delay_s,
                timer.elapsed_high_s,
                timer.elapsed_low_s,
            ]
            .iter()
            .all(|v| v.is_finite())
                && timer
                    .allowed_elapsed
                    .is_none_or(|e| e.high_s.is_finite() && e.low_s.is_finite());
            if !finite
                || timer.delay_s < 0.0
                || timer.allowed_elapsed_overflow == timer.allowed_elapsed.is_some()
            {
                return Err(StructureError::input(StructureInputErrorKind::Definition,"imported control timer requires finite original operands and coherent threshold overflow tag").at(context.slots[lane]));
            }
        }
    }
    let value = |f: fn(&ControlTimerWitness) -> f64| {
        context.tensor(
            &timers
                .iter()
                .map(|timer| timer.as_ref().map_or(0.0, f))
                .collect::<Vec<_>>(),
        )
    };
    Ok(TensorControlTimer {
        active: context.tensor(
            &timers
                .iter()
                .map(|timer| if timer.is_some() { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
        )?,
        started: value(|t| t.started_at_s)?,
        evaluated: value(|t| t.evaluated_at_s)?,
        delay: value(|t| t.delay_s)?,
        elapsed_high: value(|t| t.elapsed_high_s)?,
        elapsed_low: value(|t| t.elapsed_low_s)?,
        ready: value(|t| if t.ready { 1.0 } else { 0.0 })?,
        allowed_high: value(|t| t.allowed_elapsed.map_or(0.0, |e| e.high_s))?,
        allowed_low: value(|t| t.allowed_elapsed.map_or(0.0, |e| e.low_s))?,
        overflow: value(|t| if t.allowed_elapsed_overflow { 1.0 } else { 0.0 })?,
        within: value(|t| if t.within_lateness { 1.0 } else { 0.0 })?,
    })
}
