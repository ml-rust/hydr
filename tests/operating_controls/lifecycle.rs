use super::{fixture::*, reference_case};
use hydr::structures::*;

#[test]
fn endpoint_commands_apply_to_the_right_interval_only() -> Result {
    let batch = batch("inclusive_thresholds_and_band_hold", 1)?;
    let before = batch.snapshot_data()?;
    let trial = batch.trial_to(1.0, &[observe(1.0, 2.0, 0)])?;
    let diagnostics = trial.diagnostics()?;
    assert_eq!(diagnostics.left_commands[0].latch, ControlLatch::Low);
    assert_eq!(
        diagnostics.left_commands[0].command,
        ControlCommand::PumpStoppedIsolated
    );
    assert_eq!(diagnostics.left_commands[0].time_s, 0.0);
    assert_eq!(diagnostics.right_commands[0].latch, ControlLatch::High);
    assert_eq!(
        diagnostics.right_commands[0].command,
        ControlCommand::PumpRunning { speed_ratio: 1.0 }
    );
    assert_eq!(diagnostics.right_commands[0].time_s, 1.0);
    assert_eq!(batch.snapshot_data()?, before);
    let next = trial.into_next();
    assert_eq!(next.commands()?, diagnostics.right_commands);
    assert_eq!(next.time(), 1.0);
    Ok(())
}

#[test]
fn predecision_partitions_preserve_timer_anchors_and_future_command_history() -> Result {
    let reference = case("positive_delays_exact_expiry")?;
    let mut direct = reference_case::build(reference, 1)?;
    let mut partitioned = reference_case::build(reference, 1)?;
    partitioned = partitioned.trial_to(0.25, &[])?.into_next();
    partitioned = partitioned.trial_to(0.5, &[])?.into_next();
    assert!(
        partitioned
            .last_step()?
            .ok_or("Missing partition diagnostic")?
            .decisions
            .is_empty()
    );
    for expected in reference.decisions {
        let sample = reference_case::sample(expected.sample, 0);
        direct = direct
            .trial_to(expected.sample.observed, std::slice::from_ref(&sample))?
            .into_next();
        partitioned = partitioned
            .trial_to(expected.sample.observed, std::slice::from_ref(&sample))?
            .into_next();
        assert_eq!(partitioned.states()?, direct.states()?);
        assert_eq!(partitioned.commands()?, direct.commands()?);
        assert_eq!(partitioned.time().to_bits(), direct.time().to_bits());
        assert_eq!(partitioned.accepted_steps(), direct.accepted_steps() + 2);
        assert_eq!(partitioned.next_event(), direct.next_event());
    }
    Ok(())
}

#[test]
fn rejected_and_discarded_trials_leave_complete_accepted_memory_unchanged() -> Result {
    let batch = batch("positive_delays_exact_expiry", 1)?;
    let before = batch.snapshot_data()?;
    let candidate = batch.trial_to(1.0, &[observe(1.0, 2.0, 0)])?;
    assert!(candidate.states()?[0].pending.is_some());
    drop(candidate);
    assert_eq!(batch.snapshot_data()?, before);
    let mut future = observe(1.0, 2.0, 0);
    future.available_at_s = 2.0;
    let error = rejected(batch.trial_to(1.0, &[future]))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::UnsupportedRegime)
    );
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn explicit_terminal_support_preserves_memory_without_creating_a_decision() -> Result {
    let source = batch("inclusive_thresholds_and_band_hold", 1)?;
    let context = StructureBatch::new(source.elements().to_vec())?;
    let schedule = ControlDecisionSchedule {
        initial_time_s: 0.0,
        decision_times_s: vec![],
        end_s: 1.0,
    };
    let mut timing = source.timing().clone();
    timing.maximum_decision_gap_s = 1.0;
    let batch = context.with_operating_controls(
        source.definitions().to_vec(),
        source.initial_states().to_vec(),
        schedule,
        timing,
    )?;
    assert_eq!(
        batch.next_event(),
        Some(ControlEvent {
            time_s: 1.0,
            decision_index: None,
            terminal: true
        })
    );
    let before = batch.states()?;
    let trial = batch.trial_to(1.0, &[])?;
    let diagnostics = trial.diagnostics()?;
    assert!(diagnostics.decisions.is_empty());
    assert_eq!(diagnostics.decision_index, None);
    assert_eq!(trial.states()?, before);
    let next = trial.into_next();
    assert!(next.next_event().is_none());
    assert!(next.trial_to(2.0, &[]).is_err());
    Ok(())
}
