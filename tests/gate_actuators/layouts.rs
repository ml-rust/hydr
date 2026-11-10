use super::{fixture::*, reference_context};
use hydr::structures::*;

#[test]
fn scalar_vector_and_tail_batches_preserve_positions_and_complete_restart_data() -> Result {
    for width in reference_context::LAYOUTS {
        let batch = frozen("dyadic_open_close_hold", "O", width)?;
        let batch = walk(batch, 1.0, &[])?;
        let outputs = batch.outputs()?;
        assert_eq!(outputs.len(), width);
        for (lane, output) in outputs.iter().enumerate() {
            assert_eq!(output.element, lane);
            assert_eq!(output.position.opening_m, 0.25);
            assert!(output.position.lower_opening_m <= 0.25);
            assert!(output.position.upper_opening_m >= 0.25);
        }
        let mut restored = frozen("dyadic_open_close_hold", "O", width)?;
        restored.restore_data(&batch.snapshot_data()?)?;
        assert_eq!(restored.snapshot_data()?, batch.snapshot_data()?);
        let batch = walk(batch, 4.0, &[])?;
        let restored = walk(restored, 4.0, &[])?;
        assert_eq!(restored.snapshot_data()?, batch.snapshot_data()?);
        for state in restored.states()? {
            assert_eq!(state.position.opening_m, 1.0);
            assert_eq!(state.arrival_count, 1);
        }
    }
    Ok(())
}

#[test]
fn coincident_arrival_events_combine_scalar_flags_across_actual_gate_subset() -> Result {
    let context = mixed_context()?;
    let batch = context.with_gate_actuators(
        vec![
            definition("a", "gate-0", 0.5, 1.0),
            definition("b", "gate-2", 0.25, 1.0),
        ],
        vec![initial("a", 0.0, 1.0), initial("b", 0.0, 0.5)],
        GateActuatorSchedule {
            initial_time_s: 0.0,
            target_times_s: vec![],
            end_s: 3.0,
        },
    )?;
    let event = batch.next_event()?.ok_or("Missing shared arrival")?;
    assert_eq!(event.time_s, 2.0);
    assert!(event.arrival_lower_boundary);
    assert!(event.arrival_upper_boundary);
    assert!(!event.terminal);
    let trial = batch.trial_to(2.0, &[])?;
    let step = trial.diagnostics()?;
    assert_eq!(step.event, Some(event));
    assert_eq!(step.receipts.len(), 2);
    assert_eq!(step.receipts[0].element, 0);
    assert_eq!(step.receipts[1].element, 2);
    for receipt in &step.receipts {
        assert!(receipt.completed_episode);
    }
    let next = trial.into_next();
    assert_eq!(
        next.next_event()?.ok_or("Missing terminal event")?.time_s,
        3.0
    );
    Ok(())
}
