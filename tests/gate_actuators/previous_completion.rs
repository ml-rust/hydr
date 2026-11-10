use super::fixture::*;
use hydr::structures::*;

fn prior_completion_history() -> Result<GateActuatorBatch<crate::runtime::Runtime>> {
    let batch = build(0.0, 1.0, (0.25, 0.5), &[5.0], 8.0)?;
    let batch = walk(batch, 4.0, &[])?;
    let batch = walk(batch, 5.0, &[update(5.0, 0.0)])?;
    let batch = walk(batch, 5.25, &[])?;
    walk(batch, 5.5, &[])
}

fn change_rows(
    data: &mut GateActuatorSnapshotData,
    mut change: impl FnMut(&mut GateActuatorState) -> Result,
) -> Result {
    change(&mut data.states[0])?;
    let last = data.last_step.as_mut().ok_or("Missing immediate step")?;
    change(&mut last.receipts[0].before)?;
    change(&mut last.receipts[0].after)?;
    Ok(())
}

fn admitted_baseline() -> Result<(
    GateActuatorSnapshotData,
    GateActuatorBatch<crate::runtime::Runtime>,
)> {
    let data = prior_completion_history()?.snapshot_data()?;
    let state = &data.states[0];
    let change = state
        .last_target_change
        .as_ref()
        .ok_or("Missing target change")?;
    assert_eq!(state.target_change_count, 1);
    assert_eq!(state.arrival_count, 1);
    assert_eq!(change.time_s, 5.0);
    let latest = state
        .last_completion
        .as_ref()
        .ok_or("Missing historical completion")?;
    assert_eq!(latest.target_generation, 0);
    assert_eq!(latest.confirmed_at_s, 4.0);
    assert_eq!(latest.effective_upper_s, 4.0);
    assert_eq!(change.previous_completion.as_ref(), Some(latest));
    let mut destination = build(0.0, 1.0, (0.25, 0.5), &[5.0], 8.0)?;
    destination.restore_data(&data)?;
    assert_eq!(destination.snapshot_data()?, data);
    Ok((data, destination))
}

#[test]
fn immediately_previous_completion_cannot_disappear_from_retained_target_change() -> Result {
    let (mut data, mut destination) = admitted_baseline()?;
    change_rows(&mut data, |state| {
        state
            .last_target_change
            .as_mut()
            .ok_or("Missing target change")?
            .previous_completion = None;
        Ok(())
    })?;
    malformed_restore(&mut destination, &data)
}

#[test]
fn immediately_previous_completion_matches_the_same_historical_receipt_bits() -> Result {
    let (original, mut destination) = admitted_baseline()?;
    for alter_previous in [true, false] {
        let mut data = original.clone();
        change_rows(&mut data, |state| {
            let receipt = if alter_previous {
                state
                    .last_target_change
                    .as_mut()
                    .ok_or("Missing target change")?
                    .previous_completion
                    .as_mut()
                    .ok_or("Missing previous completion")?
            } else {
                state
                    .last_completion
                    .as_mut()
                    .ok_or("Missing historical completion")?
            };
            // Both times follow the analytic upper and precede the target change at five.
            receipt.confirmed_at_s = 4.5;
            assert_eq!(receipt.original_arrival.upper_s, 4.0);
            assert_eq!(receipt.effective_upper_s, 4.0);
            assert_eq!(receipt.effective_width_s.high_s, 0.0);
            assert_eq!(receipt.effective_width_s.low_s, 0.0);
            assert!(receipt.forward.is_none());
            Ok(())
        })?;
        malformed_restore(&mut destination, &data)?;
    }
    Ok(())
}
