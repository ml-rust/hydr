use super::{
    reference_case::*,
    reference_inputs::{Case, Decision, Timer},
};
use hydr::structures::*;

pub fn command(case: &Case, output: &ControlCommandOutput, time: f64, lane: usize) -> Result {
    if output.element != lane
        || output.controller_id != controller(lane)
        || output.structure_id != device(lane)
        || output.time_s.to_bits() != time.to_bits()
        || output.command != selected(case, output.latch)
    {
        return Err(format!(
            "{} lane={lane}: command metadata or mapping differs: {output:?}",
            case.id
        )
        .into());
    }
    Ok(())
}

pub fn state(case: &Case, expected: &Decision, actual: &ControlState, lane: usize) -> Result {
    if actual.controller_id != controller(lane)
        || actual.latch != expected.latch
        || actual.pending != expected.pending
        || actual.transition_count != expected.transition_count
    {
        return Err(format!("{} lane={lane}: sampled state differs: {actual:?}", case.id).into());
    }
    Ok(())
}

fn timer(actual: &ControlTimerWitness, expected: &Timer, time: f64) -> Result {
    if actual.started_at_s.to_bits() != expected.started.to_bits()
        || actual.evaluated_at_s.to_bits() != time.to_bits()
        || actual.delay_s.to_bits() != expected.delay.to_bits()
        || actual.elapsed_high_s != expected.high
        || actual.elapsed_low_s != expected.low
        || actual.ready != expected.ready
        || actual
            .allowed_elapsed
            .map(|value| (value.high_s, value.low_s))
            != expected.allowed
        || actual.allowed_elapsed_overflow != expected.overflow
        || actual.within_lateness != expected.within
    {
        return Err(
            format!("Timer witness differs from original exact reference: {actual:?}").into(),
        );
    }
    Ok(())
}

pub fn decision(
    case: &Case,
    expected: &Decision,
    actual: &ControlDecisionReceipt,
    lane: usize,
) -> Result {
    if actual.element != lane
        || actual.controller_id != controller(lane)
        || actual.sample != sample(expected.sample, lane)
        || actual.qualifies != expected.qualifies
        || actual.transitioned != expected.transitioned
    {
        return Err(format!(
            "{} lane={lane}: decision receipt differs: {actual:?}",
            case.id
        )
        .into());
    }
    state(case, expected, &actual.after, lane)?;
    match (&actual.timer, &expected.timer) {
        (Some(actual), Some(timer_expected)) => {
            timer(actual, timer_expected, expected.sample.observed)?
        }
        (None, None) => {}
        _ => return Err(format!("{} lane={lane}: timer presence differs", case.id).into()),
    }
    Ok(())
}
