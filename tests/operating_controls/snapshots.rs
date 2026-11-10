use super::fixture::*;
use hydr::structures::*;

fn modify_transition_sample(state: &mut ControlState, sample: &ControlSample) {
    if let Some(receipt) = &mut state.last_transition
        && receipt.sample == *sample
    {
        receipt.sample.stage_m = 3.0;
    }
}

#[test]
fn latest_transition_sample_remains_identical_to_the_latest_observation_after_time_only_steps()
-> Result {
    let source = accept(batch("inclusive_thresholds_and_band_hold", 1)?, 1.0, 2.0)?;
    let source = source.trial_to(1.5, &[])?.into_next();
    let mut data = source.snapshot_data()?;
    let original = data
        .states
        .first()
        .and_then(|state| state.last_sample.clone())
        .ok_or("Missing latest observation")?;
    let transition = data
        .states
        .first()
        .and_then(|state| state.last_transition.as_ref())
        .ok_or("Missing latest transition")?;
    assert_eq!(transition.sample, original);
    assert_eq!(transition.decision_index, 0);
    for state in &mut data.states {
        modify_transition_sample(state, &original);
    }
    if let Some(step) = &mut data.last_step {
        for decision in &mut step.decisions {
            modify_transition_sample(&mut decision.before, &original);
            modify_transition_sample(&mut decision.after, &original);
        }
    }
    assert_eq!(data.states[0].last_sample.as_ref(), Some(&original));
    assert_eq!(
        data.states[0]
            .last_transition
            .as_ref()
            .ok_or("Missing altered transition")?
            .sample
            .stage_m,
        3.0
    );
    let mut destination = batch("inclusive_thresholds_and_band_hold", 1)?;
    malformed_restore(&mut destination, &data)
}

#[test]
fn processed_decisions_and_a_later_time_only_interval_require_distinct_accepted_steps() -> Result {
    let source = accept(batch("inclusive_thresholds_and_band_hold", 1)?, 1.0, 2.0)?;
    let source = accept(source, 2.0, 1.5)?;
    let source = source.trial_to(2.5, &[])?.into_next();
    let mut data = source.snapshot_data()?;
    assert_eq!(data.next_decision_index, 2);
    assert_eq!(data.accepted_steps, 3);
    assert!(
        data.last_step
            .as_ref()
            .ok_or("Missing time-only interval")?
            .start_s
            > data.schedule.initial_time_s
    );
    data.accepted_steps = 2;
    let mut destination = batch("inclusive_thresholds_and_band_hold", 1)?;
    malformed_restore(&mut destination, &data)
}
