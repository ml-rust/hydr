use super::fixture::*;
use hydr::structures::*;

fn completed_history() -> Result<GateActuatorBatch<crate::runtime::Runtime>> {
    let batch = build(0.0, 1.0, (0.25, 0.5), &[5.0, 6.0], 9.0)?;
    let batch = walk(batch, 4.0, &[])?;
    let batch = walk(batch, 5.0, &[update(5.0, 0.0)])?;
    let batch = walk(batch, 6.0, &[update(6.0, 0.0)])?;
    let batch = walk(batch, 7.0, &[])?;
    walk(batch, 8.0, &[])
}

fn replace_current(data: &mut GateActuatorSnapshotData, state: GateActuatorState) -> Result {
    data.states[0] = state.clone();
    data.last_step.as_mut().ok_or("Missing last step")?.receipts[0].after = state;
    Ok(())
}

#[test]
fn trusted_and_caller_snapshots_restart_complete_future_target_and_arrival_history() -> Result {
    let mut source = build(0.0, 1.0, (0.25, 0.5), &[5.0, 6.0], 9.0)?;
    for (end, target) in [
        (1.0, None),
        (4.0, None),
        (5.0, Some(0.0)),
        (6.0, Some(0.0)),
        (7.0, None),
        (8.0, None),
        (9.0, None),
    ] {
        let trusted = source.snapshot();
        let data = source.snapshot_data()?;
        assert_eq!(trusted.time().to_bits(), source.time().to_bits());
        assert_eq!(trusted.data()?, data);
        let mut restored = build(0.0, 1.0, (0.25, 0.5), &[5.0, 6.0], 9.0)?;
        let mut imported = restored.clone();
        restored.restore(&trusted)?;
        imported.restore_data(&data)?;
        assert_eq!(restored.snapshot_data()?, data);
        assert_eq!(imported.snapshot_data()?, data);
        let packets = target
            .map(|target| vec![update(end, target)])
            .unwrap_or_default();
        source = walk(source, end, &packets)?;
        restored = walk(restored, end, &packets)?;
        imported = walk(imported, end, &packets)?;
        assert_eq!(restored.snapshot_data()?, source.snapshot_data()?);
        assert_eq!(imported.snapshot_data()?, source.snapshot_data()?);
        assert_eq!(restored.next_event()?, source.next_event()?);
    }
    Ok(())
}

#[test]
fn previous_completion_survives_new_completion_and_cold_import_without_history_replay() -> Result {
    let source = completed_history()?;
    let state = source.states()?.remove(0);
    let change = state
        .last_target_change
        .as_ref()
        .ok_or("Missing target change")?;
    let previous = change
        .previous_completion
        .as_ref()
        .ok_or("Missing previous completion")?;
    let current = state
        .last_completion
        .as_ref()
        .ok_or("Missing current completion")?;
    assert_eq!(previous.target_generation, 0);
    assert_eq!(current.target_generation, 1);
    assert_eq!(state.arrival_count, 2);
    assert_eq!(change.previous_target_opening_m, 1.0);
    let mut restored = build(0.0, 1.0, (0.25, 0.5), &[5.0, 6.0], 9.0)?;
    restored.restore_data(&source.snapshot_data()?)?;
    let source = walk(source, 9.0, &[])?;
    let restored = walk(restored, 9.0, &[])?;
    assert_eq!(restored.snapshot_data()?, source.snapshot_data()?);
    Ok(())
}

#[test]
fn forged_current_state_receipts_and_counters_reject_atomically() -> Result {
    let source = completed_history()?;
    let original = source.snapshot_data()?;
    let mut destination = build(0.0, 1.0, (0.25, 0.5), &[5.0, 6.0], 9.0)?;
    for variant in 0..20 {
        let mut data = original.clone();
        let mut state = data.states[0].clone();
        match variant {
            0 => state.position.opening_m = f64::from_bits(1),
            1 => state.position.upper_opening_m = f64::from_bits(1),
            2 => state.position.absolute_error_m = -0.0,
            3 => state.target_change_count = 3,
            4 => state.episode_count = 1,
            5 => state.arrival_count = 3,
            6 => state.active_episode_index = Some(0),
            7 => state.phase = GateActuatorPhase::Stalled,
            8 => state.anchor.time_s = 4.0,
            9 => {
                state
                    .latest_update
                    .as_mut()
                    .ok_or("Missing latest packet")?
                    .issued_at_s = 5.0
            }
            10 => {
                state
                    .latest_update
                    .as_mut()
                    .ok_or("Missing latest packet")?
                    .available_at_s = 7.0
            }
            11 => {
                state
                    .latest_update
                    .as_mut()
                    .ok_or("Missing latest packet")?
                    .target_opening_m = 0.25
            }
            12 => {
                state
                    .last_completion
                    .as_mut()
                    .ok_or("Missing completion")?
                    .target_generation = 0
            }
            13 => {
                state
                    .last_completion
                    .as_mut()
                    .ok_or("Missing completion")?
                    .effective_upper_s = 6.0
            }
            14 => {
                state
                    .last_completion
                    .as_mut()
                    .ok_or("Missing completion")?
                    .confirmed_at_s = 5.0
            }
            15 => {
                state
                    .last_completion
                    .as_mut()
                    .ok_or("Missing completion")?
                    .original_arrival
                    .target_opening_m = 0.25
            }
            16 => {
                state
                    .last_target_change
                    .as_mut()
                    .ok_or("Missing target change")?
                    .target_index = 1
            }
            17 => {
                state
                    .last_target_change
                    .as_mut()
                    .ok_or("Missing target change")?
                    .previous_completion
                    .as_mut()
                    .ok_or("Missing prior completion")?
                    .target_generation = 1
            }
            18 => {
                state
                    .last_target_change
                    .as_mut()
                    .ok_or("Missing target change")?
                    .previous_completion
                    .as_mut()
                    .ok_or("Missing prior completion")?
                    .original_arrival
                    .target_opening_m = 0.5
            }
            19 => {
                state
                    .last_target_change
                    .as_mut()
                    .ok_or("Missing target change")?
                    .previous_completion
                    .as_mut()
                    .ok_or("Missing prior completion")?
                    .confirmed_at_s = 0.0
            }
            _ => return Err("Unknown state mutation".into()),
        }
        replace_current(&mut data, state)?;
        malformed_restore(&mut destination, &data)?;
    }
    Ok(())
}

#[test]
fn incomplete_or_incompatible_import_packages_preserve_destination_context() -> Result {
    let source = completed_history()?;
    let original = source.snapshot_data()?;
    let mut destination = build(0.0, 1.0, (0.25, 0.5), &[5.0, 6.0], 9.0)?;
    for variant in 0..12 {
        let mut data = original.clone();
        match variant {
            0 => data.elements[0].endpoint_a.push_str(" altered"),
            1 => data.definitions[0].rate_provenance.push_str(" altered"),
            2 => data.definitions[0].opening_rate_m_per_s = 0.5,
            3 => data.initial_states[0].opening_m = -0.0,
            4 => data.schedule.target_times_s.push(8.5),
            5 => data.schedule.end_s = 10.0,
            6 => data.next_target_index = 1,
            7 => data.accepted_steps = 0,
            8 => data.time_s = 7.5,
            9 => data.last_step = None,
            10 => data.states.clear(),
            11 => data.last_step.as_mut().ok_or("Missing step")?.start_s = 8.0,
            _ => return Err("Unknown package mutation".into()),
        }
        malformed_restore(&mut destination, &data)?;
    }
    let mut different = build(0.0, 1.0, (0.25, 0.5), &[5.0, 6.5], 9.0)?;
    let before = different.snapshot_data()?;
    kind(
        different.restore(&source.snapshot()),
        StructureInputErrorKind::Definition,
    )?;
    assert_eq!(different.snapshot_data()?, before);
    malformed_restore(&mut different, &original)?;
    Ok(())
}

#[test]
fn prior_accepted_steps_cover_processed_packets_even_after_time_only_substep() -> Result {
    let batch = build(0.0, 0.0, (0.0, 0.0), &[1.0, 2.0, 3.0], 4.0)?;
    let batch = walk(batch, 1.0, &[update(1.0, 0.0)])?;
    let batch = walk(batch, 2.0, &[update(2.0, 0.0)])?;
    let batch = walk(batch, 3.0, &[update(3.0, 0.0)])?;
    let batch = walk(batch, 3.5, &[])?;
    let mut data = batch.snapshot_data()?;
    assert_eq!(data.accepted_steps, 4);
    assert_eq!(data.next_target_index, 3);
    data.accepted_steps = 3;
    let mut destination = build(0.0, 0.0, (0.0, 0.0), &[1.0, 2.0, 3.0], 4.0)?;
    malformed_restore(&mut destination, &data)
}

#[test]
fn unmodified_motion_receipts_restore_and_forged_fused_operands_reject() -> Result {
    let source = walk(build(0.0, 1.0, (0.3, 0.7), &[], 2.0)?, 0.4, &[])?;
    let original = source.snapshot_data()?;
    let mut destination = build(0.0, 1.0, (0.3, 0.7), &[], 2.0)?;
    destination.restore_data(&original)?;
    assert_eq!(destination.snapshot_data()?, original);
    for variant in 0..4 {
        let mut data = original.clone();
        let forward = data.last_step.as_mut().ok_or("Missing step")?.receipts[0]
            .left_forward
            .as_mut()
            .ok_or("Missing forward receipt")?;
        let motion = forward
            .center_motion
            .as_mut()
            .ok_or("Missing center motion")?;
        match variant {
            0 => motion.primary.signed_rate_m_per_s = 0.5,
            1 => motion.primary.time_component_s = 0.5,
            2 => motion.primary.result_m = 0.0,
            3 => motion.primary.rounding_error_m = 0.0,
            _ => return Err("Unknown FMA mutation".into()),
        }
        malformed_restore(&mut destination, &data)?;
    }
    Ok(())
}

#[test]
fn old_forward_completion_remains_authoritative_after_new_episode_completes() -> Result {
    let source = build(1.0, 0.0, (1.0, 0.02), &[50.0], 52.0)?;
    let source = walk(source, 50.0, &[update(50.0, 1.0)])?;
    let change = source.states()?[0]
        .last_target_change
        .clone()
        .ok_or("Missing target change")?;
    let previous = change
        .previous_completion
        .clone()
        .ok_or("Missing old forward completion")?;
    assert_eq!(previous.confirmed_at_s, 50.0);
    assert!(previous.original_arrival.upper_s > 50.0);
    assert!(previous.forward.is_some());
    let source = walk(source, 51.0, &[])?;
    let source = walk(source, 51.5, &[])?;
    let original = source.snapshot_data()?;
    assert_eq!(original.states[0].arrival_count, 2);
    assert_eq!(
        original.states[0]
            .last_target_change
            .as_ref()
            .ok_or("Missing retained change")?
            .previous_completion,
        Some(previous)
    );
    let mut destination = build(1.0, 0.0, (1.0, 0.02), &[50.0], 52.0)?;
    destination.restore_data(&original)?;
    assert_eq!(destination.snapshot_data()?, original);
    for variant in 0..3 {
        let mut data = original.clone();
        let mut state = data.states[0].clone();
        let old = &mut state
            .last_target_change
            .as_mut()
            .ok_or("Missing retained change")?
            .previous_completion;
        match variant {
            0 => *old = None,
            1 => old.as_mut().ok_or("Missing previous completion")?.forward = None,
            2 => {
                old.as_mut()
                    .ok_or("Missing previous completion")?
                    .effective_upper_s = f64::from_bits(50.0f64.to_bits() - 1)
            }
            _ => return Err("Unknown previous completion mutation".into()),
        }
        replace_current(&mut data, state)?;
        malformed_restore(&mut destination, &data)?;
    }
    let resumed = walk(destination, 52.0, &[])?;
    let source = walk(source, 52.0, &[])?;
    assert_eq!(resumed.snapshot_data()?, source.snapshot_data()?);
    Ok(())
}
