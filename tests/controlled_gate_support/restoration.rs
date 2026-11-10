use super::fixture::*;
use hydr::structures::*;

fn independently_admitted(config: &Config, data: &ControlledGateSnapshotData) -> Result {
    let context = config.context()?;
    let mut controls = config.controls(&context)?;
    let mut actors = config.actuators(&context)?;
    controls.restore_data(&data.controls)?;
    actors.restore_data(&data.actuators)?;
    let mut destination = config.build()?;
    malformed_restore(&mut destination, data)
}

#[test]
fn trusted_and_cold_restoration_preserve_pending_qualification_and_future_response() -> Result {
    let mut config = Config::switching();
    config.times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    config.end = 8.0;
    config.delays = (2.0, 1.0);
    config.rates = (0.5, 0.25);
    config.maximum_gap = 3.0;
    config.lateness = 0.0;
    for width in [1, 33] {
        config.width = width;
        let original = accept(config.build()?, 1.0, Some(2.0))?;
        let original = accept(original, 1.5, None)?;
        let trusted = original.snapshot();
        let data = original.snapshot_data()?;
        assert_eq!(trusted.time().to_bits(), 1.5_f64.to_bits());
        assert_eq!(trusted.data()?, data);
        let mut restored = config.build()?;
        let mut imported = config.build()?;
        restored.restore(&trusted)?;
        imported.restore_data(&data)?;
        assert_eq!(restored.snapshot_data()?, data);
        assert_eq!(imported.snapshot_data()?, data);
        let mut original = original;
        for (time, stage) in [
            (2.0, Some(2.0)),
            (3.0, Some(2.0)),
            (4.0, Some(1.0)),
            (5.0, Some(1.0)),
            (6.0, None),
            (8.0, None),
        ] {
            original = accept(original, time, stage)?;
            restored = accept(restored, time, stage)?;
            imported = accept(imported, time, stage)?;
            assert_eq!(restored.snapshot_data()?, original.snapshot_data()?);
            assert_eq!(imported.snapshot_data()?, original.snapshot_data()?);
        }
    }
    Ok(())
}

#[test]
fn discarded_pair_trials_and_rewound_decisions_preserve_complete_accepted_values() -> Result {
    let config = Config::switching();
    let mut batch = config.build()?;
    let before = batch.snapshot_data()?;
    drop(batch.trial_to(1.0, &samples(1, 1.0, 2.0))?);
    assert_eq!(batch.snapshot_data()?, before);
    let snapshot = batch.snapshot();
    batch = accept(batch, 1.0, Some(2.0))?;
    assert_eq!(batch.controls().states()?[0].transition_count, 1);
    batch.restore(&snapshot)?;
    assert_eq!(batch.snapshot_data()?, before);
    let batch = accept(batch, 1.0, Some(1.5))?;
    assert_eq!(batch.controls().states()?[0].transition_count, 0);
    assert_eq!(batch.actuators().states()?[0].target_change_count, 0);
    Ok(())
}

#[test]
fn compatible_independent_contexts_restore_both_snapshot_forms() -> Result {
    let config = Config::switching();
    let source = accept(config.build()?, 1.0, Some(2.0))?;
    let source = accept(source, 1.5, None)?;
    let mut trusted = config.build()?;
    let mut imported = config.build()?;
    trusted.restore(&source.snapshot())?;
    imported.restore_data(&source.snapshot_data()?)?;
    assert_eq!(trusted.snapshot_data()?, source.snapshot_data()?);
    assert_eq!(imported.snapshot_data()?, source.snapshot_data()?);
    Ok(())
}

#[test]
fn incompatible_child_original_bits_reject_both_snapshot_forms_atomically() -> Result {
    let config = Config::switching();
    let source = accept(config.build()?, 1.0, Some(2.0))?;
    let mut changed = config.clone();
    changed.rates.0 = 0.5;
    let mut destination = changed.build()?;
    let before = destination.snapshot_data()?;
    let error = rejected(destination.restore(&source.snapshot()))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition)
    );
    assert_eq!(destination.snapshot_data()?, before);
    malformed_restore(&mut destination, &source.snapshot_data()?)
}

#[test]
fn individually_admitted_child_clocks_and_interval_counts_must_match() -> Result {
    let config = Config::switching();
    let source = accept(config.build()?, 1.0, Some(2.0))?;
    let source = accept(source, 1.25, None)?;
    let original = source.snapshot_data()?;
    let context = config.context()?;
    let actor = config.actuators(&context)?;
    let mut data = original.clone();
    data.actuators = actor.trial_to(0.5, &[])?.into_next().snapshot_data()?;
    independently_admitted(&config, &data)?;
    let actor = actor.trial_to(0.5, &[])?.into_next();
    let actor = actor.trial_to(1.0, &updates(1, 1.0, 1.0))?.into_next();
    data.actuators = actor.trial_to(1.25, &[])?.into_next().snapshot_data()?;
    assert_eq!(data.controls.time_s, data.actuators.time_s);
    assert_ne!(data.controls.accepted_steps, data.actuators.accepted_steps);
    independently_admitted(&config, &data)
}

#[test]
fn individually_admitted_last_intervals_must_share_the_original_start() -> Result {
    let config = Config::switching();
    let source = accept(config.build()?, 0.25, None)?;
    let source = accept(source, 1.0, Some(2.0))?;
    let source = accept(source, 1.25, None)?;
    let context = config.context()?;
    let actor = config.actuators(&context)?;
    let actor = actor.trial_to(1.0, &updates(1, 1.0, 1.0))?.into_next();
    let actor = actor.trial_to(1.125, &[])?.into_next();
    let actor = actor.trial_to(1.25, &[])?.into_next();
    let mut data = source.snapshot_data()?;
    data.actuators = actor.snapshot_data()?;
    assert_eq!(data.controls.accepted_steps, data.actuators.accepted_steps);
    assert_eq!(data.controls.time_s, data.actuators.time_s);
    assert_ne!(
        data.controls
            .last_step
            .as_ref()
            .ok_or("Missing control step")?
            .start_s,
        data.actuators
            .last_step
            .as_ref()
            .ok_or("Missing actuator step")?
            .start_s
    );
    independently_admitted(&config, &data)
}

#[test]
fn individually_admitted_current_target_and_packet_availability_must_match_commands() -> Result {
    let config = Config::switching();
    let source = accept(config.build()?, 1.0, Some(2.0))?;
    let context = config.context()?;
    for variant in 0..2 {
        let mut packets = updates(1, 1.0, 1.0);
        if variant == 0 {
            packets[0].target_opening_m = 0.5;
        } else {
            packets[0].available_at_s = 0.5;
        }
        let actor = config
            .actuators(&context)?
            .trial_to(1.0, &packets)?
            .into_next();
        let mut data = source.snapshot_data()?;
        data.actuators = actor.snapshot_data()?;
        independently_admitted(&config, &data)?;
    }
    Ok(())
}

#[test]
fn latest_target_change_must_match_the_retained_control_transition() -> Result {
    let config = Config::switching();
    let context = config.context()?;
    let high = accept(config.build()?, 1.0, Some(2.0))?;
    let repeated = accept(high, 2.0, Some(2.0))?;
    let actor = config
        .actuators(&context)?
        .trial_to(1.0, &updates(1, 1.0, 0.0))?
        .into_next();
    let actor = actor.trial_to(2.0, &updates(1, 2.0, 1.0))?.into_next();
    let mut data = repeated.snapshot_data()?;
    data.actuators = actor.snapshot_data()?;
    assert_eq!(
        data.controls.states[0].transition_count,
        data.actuators.states[0].target_change_count
    );
    independently_admitted(&config, &data)?;

    let source = accept(config.build()?, 1.0, Some(2.0))?;
    let source = accept(source, 2.0, Some(1.0))?;
    let actor = config
        .actuators(&context)?
        .trial_to(1.0, &updates(1, 1.0, 0.5))?
        .into_next();
    let actor = actor.trial_to(2.0, &updates(1, 2.0, 0.0))?.into_next();
    let mut data = source.snapshot_data()?;
    data.actuators = actor.snapshot_data()?;
    assert_eq!(
        data.controls.states[0].transition_count,
        data.actuators.states[0].target_change_count
    );
    independently_admitted(&config, &data)
}

#[test]
fn equal_target_packets_require_selected_literal_bits_in_current_and_prior_states() -> Result {
    let mut config = Config::switching();
    config.target = -0.0;
    config.low = -0.0;
    config.high = 0.0;
    config.rates = (0.0, 0.0);
    let context = config.context()?;
    let high = accept(config.build()?, 1.0, Some(2.0))?;
    let wrong_actor = config
        .actuators(&context)?
        .trial_to(1.0, &updates(1, 1.0, -0.0))?
        .into_next();
    let mut current = high.snapshot_data()?;
    current.actuators = wrong_actor.snapshot_data()?;
    independently_admitted(&config, &current)?;

    let low = accept(high, 2.0, Some(1.0))?;
    let wrong_actor = wrong_actor
        .trial_to(2.0, &updates(1, 2.0, -0.0))?
        .into_next();
    let mut prior = low.snapshot_data()?;
    prior.actuators = wrong_actor.snapshot_data()?;
    assert_eq!(prior.controls.states[0].transition_count, 2);
    assert_eq!(prior.actuators.states[0].target_change_count, 0);
    assert_eq!(
        prior.actuators.states[0]
            .latest_update
            .as_ref()
            .ok_or("Missing latest packet")?
            .target_opening_m
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    independently_admitted(&config, &prior)
}

#[test]
fn incomplete_pair_packages_reject_without_partially_replacing_either_child() -> Result {
    let config = Config::switching();
    let source = accept(config.build()?, 1.0, Some(2.0))?;
    let original = source.snapshot_data()?;
    let mut destination = accept(config.build()?, 1.0, Some(1.5))?;
    for variant in 0..4 {
        let mut data = original.clone();
        match variant {
            0 => data.controls.states.clear(),
            1 => data.actuators.states.clear(),
            2 => data.controls.last_step = None,
            3 => data.actuators.last_step = None,
            _ => return Err("Unknown pair package variant".into()),
        }
        malformed_restore(&mut destination, &data)?;
    }
    Ok(())
}
