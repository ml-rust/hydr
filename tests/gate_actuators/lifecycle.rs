use super::fixture::*;
use hydr::structures::*;

#[test]
fn directional_rates_move_actual_opening_and_reversal_starts_from_left_position() -> Result {
    let initial = build(0.25, 1.0, (0.5, 0.125), &[1.0, 3.0], 5.0)?;
    let first = walk(initial, 0.5, &[])?;
    assert_eq!(position(&first, 0.5)?.position.opening_m, 0.5);
    let trial = first.trial_to(1.0, &[update(1.0, 0.0)])?;
    let step = trial.diagnostics()?;
    assert_eq!(step.receipts[0].left_position.opening_m, 0.75);
    assert_eq!(step.receipts[0].after.target_opening_m, 0.0);
    assert_eq!(
        step.receipts[0].after.anchor.position,
        step.receipts[0].left_position
    );
    let reversed = walk(trial.into_next(), 2.0, &[])?;
    assert_eq!(position(&reversed, 0.625)?.position.opening_m, 0.625);
    let changed = walk(reversed, 3.0, &[update(3.0, 0.5)])?;
    let state = changed.states()?.remove(0);
    assert_eq!(state.position.opening_m, 0.5);
    assert_eq!(state.episode_count, 2);
    assert_eq!(state.arrival_count, 0);
    assert_eq!(state.active_episode_index, None);
    Ok(())
}

#[test]
fn coincident_arrival_completes_old_episode_before_new_right_target() -> Result {
    let batch = build(0.0, 1.0, (0.5, 0.25), &[2.0], 6.0)?;
    let next = walk(batch, 2.0, &[update(2.0, 0.0)])?;
    let step = next.last_step()?.ok_or("Missing endpoint diagnostic")?;
    assert_eq!(step.receipts[0].left_position.opening_m, 1.0);
    assert!(step.receipts[0].completed_episode);
    assert!(step.receipts[0].target_changed);
    let state = next.states()?.remove(0);
    assert_eq!(state.position.opening_m, 1.0);
    assert_eq!(state.target_opening_m, 0.0);
    assert_eq!(state.arrival_count, 1);
    assert_eq!(state.episode_count, 2);
    assert_eq!(state.target_change_count, 1);
    let later = walk(next, 3.0, &[])?;
    assert_eq!(position(&later, 0.75)?.position.opening_m, 0.75);
    let closed = walk(later, 6.0, &[])?;
    assert_eq!(
        position(&closed, 0.0)?.support,
        GateActuatorSupport::ExactClosed
    );
    assert_eq!(closed.states()?[0].arrival_count, 2);
    assert!(closed.next_event()?.is_none());
    Ok(())
}

#[test]
fn repeated_targets_preserve_anchor_and_arrival_but_retain_latest_causal_packet() -> Result {
    let batch = build(0.0, 1.0, (0.25, 0.5), &[1.0, 2.0, 4.0], 5.0)?;
    let original = batch.states()?.remove(0);
    let mut batch = batch;
    for time in [1.0, 2.0, 4.0] {
        let mut packet = update(time, 1.0);
        packet.available_at_s = -1.0;
        batch = walk(batch, time, std::slice::from_ref(&packet))?;
        let state = batch.states()?.remove(0);
        assert_eq!(state.anchor, original.anchor);
        assert_eq!(state.arrival, original.arrival);
        assert_eq!(state.target_change_count, 0);
        assert_eq!(state.episode_count, 1);
        assert_eq!(state.latest_update, Some(packet));
    }
    assert_eq!(batch.states()?[0].arrival_count, 1);
    let batch = walk(batch, 5.0, &[])?;
    assert_eq!(batch.states()?[0].arrival_count, 1);
    Ok(())
}

#[test]
fn signed_zero_repeats_preserve_active_target_bits_and_original_initial_inputs() -> Result {
    let batch = build(0.0, -0.0, (0.0, 0.0), &[1.0], 2.0)?;
    let original = batch.states()?.remove(0);
    assert_eq!(batch.initial_states()[0].opening_m.to_bits(), 0);
    assert_eq!(original.position.opening_m.to_bits(), (-0.0f64).to_bits());
    assert_eq!(original.episode_count, 0);
    let packet = update(1.0, 0.0);
    let batch = walk(batch, 1.0, std::slice::from_ref(&packet))?;
    let state = batch.states()?.remove(0);
    assert_eq!(state.target_opening_m.to_bits(), (-0.0f64).to_bits());
    assert_eq!(state.anchor, original.anchor);
    assert_eq!(
        state
            .latest_update
            .ok_or("Missing repeated packet")?
            .target_opening_m
            .to_bits(),
        0
    );
    assert_eq!(state.arrival_count, 0);
    assert_eq!(state.target_change_count, 0);
    Ok(())
}

#[test]
fn zero_directional_rate_stalls_without_false_completion_or_unused_division() -> Result {
    let batch = build(0.25, 1.0, (0.0, 0.5), &[2.0, 3.0], 5.0)?;
    assert_eq!(batch.states()?[0].phase, GateActuatorPhase::Stalled);
    assert_eq!(
        batch.next_event()?.ok_or("Missing target event")?.time_s,
        2.0
    );
    let batch = walk(batch, 1.0, &[])?;
    assert_eq!(position(&batch, 0.25)?.position.opening_m, 0.25);
    assert_eq!(batch.states()?[0].arrival_count, 0);
    let batch = walk(batch, 2.0, &[update(2.0, 0.0)])?;
    let batch = walk(batch, 2.5, &[])?;
    assert_eq!(
        position(&batch, 0.0)?.support,
        GateActuatorSupport::ExactClosed
    );
    assert_eq!(batch.states()?[0].arrival_count, 1);
    let batch = walk(batch, 3.0, &[update(3.0, 1.0)])?;
    let batch = walk(batch, 5.0, &[])?;
    assert_eq!(batch.states()?[0].phase, GateActuatorPhase::Stalled);
    assert_eq!(batch.states()?[0].arrival_count, 1);
    Ok(())
}

#[test]
fn ordinary_substeps_keep_original_motion_anchor_and_partition_independent_position() -> Result {
    let direct = build(0.0, 1.0, (0.25, 0.5), &[], 5.0)?;
    let anchor = direct.states()?[0].anchor;
    let mut partitioned = direct.clone();
    for time in [0.125, 0.5, 1.0, 1.5] {
        partitioned = walk(partitioned, time, &[])?;
        assert_eq!(partitioned.states()?[0].anchor, anchor);
    }
    let direct = walk(direct, 2.0, &[])?;
    let partitioned = walk(partitioned, 2.0, &[])?;
    assert_eq!(partitioned.states()?, direct.states()?);
    assert_eq!(partitioned.outputs()?, direct.outputs()?);
    assert_eq!(partitioned.accepted_steps(), direct.accepted_steps() + 4);
    Ok(())
}

#[test]
fn rejected_and_discarded_trials_preserve_every_accepted_receipt() -> Result {
    let batch = build(0.0, 1.0, (0.25, 0.5), &[2.0], 5.0)?;
    let before = batch.snapshot_data()?;
    drop(batch.trial_to(1.0, &[])?);
    assert_eq!(batch.snapshot_data()?, before);
    let mut future = update(2.0, 0.0);
    future.available_at_s = 3.0;
    kind(
        batch.trial_to(2.0, &[future]),
        StructureInputErrorKind::UnsupportedRegime,
    )?;
    assert_eq!(batch.snapshot_data()?, before);
    kind(
        batch.trial_to(3.0, &[]),
        StructureInputErrorKind::UnsupportedRegime,
    )?;
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}
