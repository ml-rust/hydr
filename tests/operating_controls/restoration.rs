use super::{fixture::*, reference_case};
use hydr::structures::*;

#[test]
fn trusted_and_caller_snapshots_restart_identical_supplied_histories() -> Result {
    for id in [
        "positive_delays_exact_expiry",
        "disqualification_resets_pending",
    ] {
        let reference = case(id)?;
        for width in [1, 33] {
            let mut original = reference_case::build(reference, width)?;
            original = original.trial_to(0.5, &[])?.into_next();
            for (index, expected) in reference.decisions.iter().enumerate() {
                let trusted = original.snapshot();
                let data = original.snapshot_data()?;
                assert_eq!(trusted.time().to_bits(), original.time().to_bits());
                assert_eq!(trusted.data()?, data);
                let mut restored = reference_case::build(reference, width)?;
                let mut imported = reference_case::build(reference, width)?;
                restored.restore(&trusted)?;
                imported.restore_data(&data)?;
                assert_eq!(restored.snapshot_data()?, data);
                assert_eq!(imported.snapshot_data()?, data);
                for future in &reference.decisions[index..] {
                    let samples = (0..width)
                        .map(|lane| reference_case::sample(future.sample, lane))
                        .collect::<Vec<_>>();
                    restored = restored
                        .trial_to(future.sample.observed, &samples)?
                        .into_next();
                    imported = imported
                        .trial_to(future.sample.observed, &samples)?
                        .into_next();
                    assert_eq!(imported.snapshot_data()?, restored.snapshot_data()?);
                }
                let samples = (0..width)
                    .map(|lane| reference_case::sample(expected.sample, lane))
                    .collect::<Vec<_>>();
                original = original
                    .trial_to(expected.sample.observed, &samples)?
                    .into_next();
            }
            let mut destination = reference_case::build(reference, width)?;
            destination.restore_data(&original.snapshot_data()?)?;
            assert_eq!(destination.snapshot_data()?, original.snapshot_data()?);
        }
    }
    Ok(())
}

#[test]
fn incompatible_original_bits_and_context_reject_both_snapshot_forms_atomically() -> Result {
    let source = accept(batch("inclusive_thresholds_and_band_hold", 1)?, 1.0, 2.0)?;
    let context = StructureBatch::new(source.elements().to_vec())?;
    let mut definitions = source.definitions().to_vec();
    definitions[0].high_qualification_delay_s = -0.0;
    let mut destination = context.with_operating_controls(
        definitions,
        source.initial_states().to_vec(),
        source.schedule().clone(),
        source.timing().clone(),
    )?;
    assert_ne!(
        destination.definitions()[0]
            .high_qualification_delay_s
            .to_bits(),
        source.definitions()[0].high_qualification_delay_s.to_bits()
    );
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
fn incomplete_or_contradictory_imports_preserve_the_complete_destination() -> Result {
    let source = accept(batch("inclusive_thresholds_and_band_hold", 1)?, 1.0, 2.0)?;
    let original = source.snapshot_data()?;
    let mut destination = accept(batch("inclusive_thresholds_and_band_hold", 1)?, 1.0, 1.5)?;
    for variant in 0..18 {
        let mut data = original.clone();
        match variant {
            0 => data.definitions[0].provenance.push_str(" altered"),
            1 => data.elements[0].endpoint_a.push_str(" altered"),
            2 => data.schedule.end_s += 1.0,
            3 => data.timing.sampled_applicability.push_str(" altered"),
            4 => data.initial_states[0].last_transition_at_s = Some(-1.0),
            5 => data.last_step = None,
            6 => data.time_s = 0.5,
            7 => data.next_decision_index = 0,
            8 => data.accepted_steps = 0,
            9 => data.states[0].controller_id.push_str(" altered"),
            10 => {
                data.states[0].pending = Some(ControlPending {
                    target: ControlLatch::High,
                    since_s: 1.0,
                })
            }
            11 => {
                data.states[0]
                    .last_sample
                    .as_mut()
                    .ok_or("Missing sample")?
                    .available_at_s = 2.0
            }
            12 => data.states[0].last_sample = None,
            13 => data.states[0].transition_count = 2,
            14 => data.states[0].last_transition = None,
            15 => {
                data.states[0]
                    .last_transition
                    .as_mut()
                    .ok_or("Missing transition")?
                    .decision_index = 1
            }
            16 => {
                data.last_step
                    .as_mut()
                    .ok_or("Missing step")?
                    .right_commands[0]
                    .time_s = 0.5
            }
            17 => data.last_step.as_mut().ok_or("Missing step")?.decisions[0].qualifies = false,
            _ => return Err("Unknown import mutation".into()),
        }
        malformed_restore(&mut destination, &data)?;
    }
    Ok(())
}

#[test]
fn pending_and_retained_transition_receipts_survive_time_only_restart() -> Result {
    let pending = accept(batch("positive_delays_exact_expiry", 1)?, 1.0, 2.0)?;
    let pending = pending.trial_to(1.5, &[])?.into_next();
    let data = pending.snapshot_data()?;
    assert_eq!(
        data.states[0]
            .pending
            .ok_or("Missing pending timer")?
            .since_s
            .to_bits(),
        1.0_f64.to_bits()
    );
    let mut restored = batch("positive_delays_exact_expiry", 1)?;
    restored.restore_data(&data)?;
    assert_eq!(restored.snapshot_data()?, data);
    let transitioned = accept(batch("inclusive_thresholds_and_band_hold", 1)?, 1.0, 2.0)?;
    let transitioned = transitioned.trial_to(1.5, &[])?.into_next();
    let data = transitioned.snapshot_data()?;
    assert_eq!(
        data.states[0]
            .last_transition
            .as_ref()
            .ok_or("Missing retained transition")?
            .time_s
            .to_bits(),
        1.0_f64.to_bits()
    );
    let mut restored = batch("inclusive_thresholds_and_band_hold", 1)?;
    restored.restore_data(&data)?;
    assert_eq!(restored.snapshot_data()?, data);
    Ok(())
}
