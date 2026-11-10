use super::fixture::*;
use hydr::structures::*;

fn interrupted_history() -> Result<GateActuatorBatch> {
    let batch = build(0.0, 0.0, (1.0, 1.0), &[1.0, 4.0, 4.125], 10.0)?;
    let batch = walk(batch, 1.0, &[update(1.0, 1.0)])?;
    let batch = walk(batch, 2.0, &[])?;
    let batch = walk(batch, 4.0, &[update(4.0, 0.75)])?;
    let batch = walk(batch, 4.125, &[update(4.125, 0.0)])?;
    walk(batch, 4.25, &[])
}

fn replace_historical(
    data: &mut GateActuatorSnapshotData,
    receipt: &GateActuatorCompletionReceipt,
) -> Result {
    data.states[0].last_completion = Some(receipt.clone());
    let last = data.last_step.as_mut().ok_or("Missing immediate step")?;
    last.receipts[0].before.last_completion = Some(receipt.clone());
    last.receipts[0].after.last_completion = Some(receipt.clone());
    Ok(())
}

#[test]
fn historical_completion_target_must_remain_inside_actual_gate_range() -> Result {
    let source = interrupted_history()?;
    let mut data = source.snapshot_data()?;
    let mut destination = build(0.0, 0.0, (1.0, 1.0), &[1.0, 4.0, 4.125], 10.0)?;
    destination.restore_data(&data)?;
    assert_eq!(destination.snapshot_data()?, data);
    let state = &data.states[0];
    assert_eq!(state.target_change_count, 3);
    assert_eq!(state.arrival_count, 1);
    assert!(
        state
            .last_target_change
            .as_ref()
            .ok_or("Missing latest change")?
            .previous_completion
            .is_none()
    );
    let mut forged = state
        .last_completion
        .clone()
        .ok_or("Missing historical completion")?;
    assert_eq!(forged.target_generation, 1);
    assert_eq!(forged.episode_index, 0);
    assert_eq!(forged.original_arrival.anchor.time_s, 1.0);
    assert_eq!(forged.original_arrival.anchor.position.opening_m, 0.0);
    // The forged rate-one analytic receipt is self-consistent but exceeds the actual maximum of one metre.
    forged.original_arrival.target_opening_m = 2.0;
    forged.original_arrival.lower_s = 3.0;
    forged.original_arrival.upper_s = 3.0;
    forged.original_arrival.width_s = GateActuatorTimeExpansion {
        high_s: 0.0,
        low_s: 0.0,
    };
    forged.confirmed_at_s = 3.0;
    forged.effective_upper_s = 3.0;
    forged.effective_width_s = GateActuatorTimeExpansion {
        high_s: 0.0,
        low_s: 0.0,
    };
    forged.forward = None;
    replace_historical(&mut data, &forged)?;
    malformed_restore(&mut destination, &data)
}

fn later_interrupted_history() -> Result<GateActuatorBatch> {
    walk(interrupted_history()?, 4.375, &[])
}

#[test]
fn historical_completion_confirmation_precedes_current_target_activation() -> Result {
    let source = later_interrupted_history()?;
    let mut data = source.snapshot_data()?;
    let mut destination = build(0.0, 0.0, (1.0, 1.0), &[1.0, 4.0, 4.125], 10.0)?;
    destination.restore_data(&data)?;
    assert_eq!(destination.snapshot_data()?, data);
    assert_eq!(data.states[0].anchor.time_s, 4.125);
    assert_eq!(data.last_step.as_ref().ok_or("Missing step")?.start_s, 4.25);
    let mut forged = data.states[0]
        .last_completion
        .clone()
        .ok_or("Missing historical completion")?;
    assert_eq!(forged.target_generation, 1);
    assert_eq!(forged.original_arrival.upper_s, 2.0);
    forged.confirmed_at_s = 4.2;
    // The old target reached one at time two. It cannot first complete after a newer target activates.
    replace_historical(&mut data, &forged)?;
    malformed_restore(&mut destination, &data)
}

#[test]
fn historical_completion_confirmation_precedes_the_next_generation_activation() -> Result {
    let source = later_interrupted_history()?;
    let mut data = source.snapshot_data()?;
    let mut destination = build(0.0, 0.0, (1.0, 1.0), &[1.0, 4.0, 4.125], 10.0)?;
    destination.restore_data(&data)?;
    assert_eq!(destination.snapshot_data()?, data);
    let latest = data.states[0]
        .last_target_change
        .as_ref()
        .ok_or("Missing latest target change")?;
    assert_eq!(latest.previous_anchor.time_s, 4.0);
    assert_eq!(latest.new_anchor.time_s, 4.125);
    let mut forged = data.states[0]
        .last_completion
        .clone()
        .ok_or("Missing historical completion")?;
    assert_eq!(forged.target_generation, 1);
    // This confirmation predates the current anchor but follows the retained generation-two activation.
    forged.confirmed_at_s = 4.0625;
    replace_historical(&mut data, &forged)?;
    malformed_restore(&mut destination, &data)
}

#[test]
fn historical_completion_activation_matches_an_original_target_event() -> Result {
    let source = interrupted_history()?;
    let mut data = source.snapshot_data()?;
    let mut destination = build(0.0, 0.0, (1.0, 1.0), &[1.0, 4.0, 4.125], 10.0)?;
    destination.restore_data(&data)?;
    assert_eq!(destination.snapshot_data()?, data);
    let mut forged = data.states[0]
        .last_completion
        .clone()
        .ok_or("Missing historical completion")?;
    assert_eq!(forged.target_generation, 1);
    assert_eq!(forged.original_arrival.target_opening_m, 1.0);
    assert_eq!(forged.original_arrival.anchor.position.opening_m, 0.0);
    forged.original_arrival.anchor.time_s = 1.5;
    forged.original_arrival.lower_s = 2.5;
    forged.original_arrival.upper_s = 2.5;
    forged.original_arrival.width_s = GateActuatorTimeExpansion {
        high_s: 0.0,
        low_s: 0.0,
    };
    forged.confirmed_at_s = 2.5;
    forged.effective_upper_s = 2.5;
    forged.effective_width_s = GateActuatorTimeExpansion {
        high_s: 0.0,
        low_s: 0.0,
    };
    forged.forward = None;
    replace_historical(&mut data, &forged)?;
    malformed_restore(&mut destination, &data)
}

#[test]
fn historical_completion_episode_index_fits_its_target_generation() -> Result {
    let source = interrupted_history()?;
    let mut data = source.snapshot_data()?;
    let mut destination = build(0.0, 0.0, (1.0, 1.0), &[1.0, 4.0, 4.125], 10.0)?;
    destination.restore_data(&data)?;
    assert_eq!(destination.snapshot_data()?, data);
    let state = &data.states[0];
    assert_eq!(state.episode_count, 3);
    assert_eq!(state.active_episode_index, Some(2));
    let mut forged = state
        .last_completion
        .clone()
        .ok_or("Missing historical completion")?;
    assert_eq!(forged.target_generation, 1);
    assert_eq!(forged.episode_index, 0);
    // Initial hold starts no episode. Generation one cannot complete episode index one.
    forged.episode_index = 1;
    replace_historical(&mut data, &forged)?;
    malformed_restore(&mut destination, &data)
}
