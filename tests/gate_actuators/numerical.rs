use super::{fixture::*, inputs as reference_inputs, reference_context};
use hydr::structures::*;

#[test]
fn ordinary_nonbinary_closure_keeps_positive_predecessor_and_certifies_zero_at_fifty() -> Result {
    let end = f64::from_bits(50.0f64.to_bits() + 1);
    let batch = build(1.0, 0.0, (0.0, 0.02), &[], end)?;
    let predecessor = f64::from_bits(50.0f64.to_bits() - 1);
    let positive = walk(batch, predecessor, &[])?;
    let output = position(&positive, f64::from_bits(0x3ca17ae147ae147b))?;
    assert!(output.position.opening_m > 0.0);
    assert_ne!(output.support, GateActuatorSupport::ExactClosed);
    let closed = walk(positive, 50.0, &[])?;
    let output = position(&closed, 0.0)?;
    assert_eq!(output.support, GateActuatorSupport::ExactClosed);
    assert_eq!(output.position.absolute_error_m.to_bits(), 0);
    assert_eq!(closed.states()?[0].arrival_count, 1);
    let receipt = closed.states()?[0]
        .last_completion
        .clone()
        .ok_or("Missing completion")?;
    assert_eq!(receipt.effective_upper_s, 50.0);
    let held = walk(closed, end, &[])?;
    assert_eq!(held.states()?[0].last_completion, Some(receipt));
    assert_eq!(held.states()?[0].arrival_count, 1);
    Ok(())
}

#[test]
fn exact_nonbinary_zero_uses_arithmetic_certificate_without_closing_its_predecessor() -> Result {
    let end = f64::from_bits(0.5f64.to_bits() + 1);
    let batch = build(0.15, 0.0, (0.0, 0.3), &[], end)?;
    let before = walk(batch, f64::from_bits(0.5f64.to_bits() - 1), &[])?;
    assert!(before.outputs()?[0].position.opening_m > 0.0);
    let closed = walk(before, 0.5, &[])?;
    let output = position(&closed, 0.0)?;
    assert_eq!(output.support, GateActuatorSupport::ExactClosed);
    assert_eq!(output.position.lower_opening_m.to_bits(), 0);
    assert_eq!(output.position.upper_opening_m.to_bits(), 0);
    Ok(())
}

#[test]
fn subnormal_positive_motion_rejects_false_zero_even_with_absolute_allowance() -> Result {
    let batch = frozen("subnormal_positive_opening", "S", 1)?;
    let before = batch.snapshot_data()?;
    for time in [0.25, 0.5] {
        let error = rejected(batch.trial_to(time, &[]))?;
        assert!(matches!(error, StructureError::Numerical { .. }), "{error}");
        assert_eq!(batch.snapshot_data()?, before);
    }
    let moved = walk(batch, 0.75, &[])?;
    assert!(moved.outputs()?[0].position.opening_m > 0.0);
    let closing = frozen("subnormal_positive_closing", "S", 1)?;
    for time in [0.5, 0.75] {
        assert!(matches!(
            rejected(closing.trial_to(time, &[]))?,
            StructureError::Numerical { .. }
        ));
    }
    let closed = walk(closing, 1.0, &[])?;
    assert_eq!(
        closed.outputs()?[0].support,
        GateActuatorSupport::ExactClosed
    );
    Ok(())
}

#[test]
fn signed_elapsed_low_preserves_positive_overflow_counterexample_or_rejects_strictly() -> Result {
    let mut strict = frozen("negative_elapsed_low_overflow_counterexample", "X", 1)?;
    let time = strict.schedule().end_s;
    while let Some(event) = strict.next_event()? {
        if event.time_s >= time {
            break;
        }
        strict = strict.trial_to(event.time_s, &[])?.into_next();
    }
    let before = strict.snapshot_data()?;
    assert!(matches!(
        rejected(strict.trial_to(time, &[]))?,
        StructureError::Numerical { .. }
    ));
    assert_eq!(strict.snapshot_data()?, before);
    let loose = frozen("negative_elapsed_low_overflow_counterexample", "X-wide", 1)?;
    let loose = walk(loose, time, &[])?;
    let output = position(&loose, 2.0f64.powi(943))?;
    assert_eq!(output.support, GateActuatorSupport::CertainlyPositive);
    assert!(output.position.lower_opening_m > 0.0);
    assert_eq!(loose.states()?[0].arrival_count, 0);
    Ok(())
}

#[test]
fn large_clock_localization_requires_explicit_window_and_never_returns_zero_time_events() -> Result
{
    let batch = frozen("large_clock_subulp_travel", "L", 1)?;
    let initial = batch.time();
    let event = batch.next_event()?.ok_or("Missing localized clock event")?;
    assert!(event.time_s > initial);
    let batch = walk(batch, event.time_s, &[])?;
    assert_eq!(position(&batch, 1.0)?.position.opening_m, 1.0);
    if let Some(next) = batch.next_event()? {
        assert!(next.time_s > batch.time());
    }
    let mut case = reference_inputs::cases()
        .into_iter()
        .find(|case| case.id == "large_clock_subulp_travel" && case.policy == "L")
        .ok_or("Missing large-clock reference")?;
    case.accuracy.maximum_arrival_window_s = 0.0;
    match reference_context::build(&case, 1) {
        Err(error) => assert!(matches!(error, StructureError::Numerical { .. }), "{error}"),
        Ok(batch) => {
            let error = rejected(batch.trial_to(event.time_s, &[]))?;
            assert!(matches!(error, StructureError::Numerical { .. }), "{error}");
        }
    }
    Ok(())
}

#[test]
fn huge_saturated_response_and_beyond_horizon_motion_skip_unused_overflow() -> Result {
    let fast = frozen("saturated_huge_product", "U", 1)?;
    let end = fast.schedule().end_s;
    let fast = walk(fast, end, &[])?;
    assert_eq!(position(&fast, 1.0)?.position.opening_m, 1.0);
    assert_eq!(fast.states()?[0].arrival_count, 1);
    let slow = frozen("slow_arrival_beyond_horizon", "B", 1)?;
    assert_eq!(slow.states()?[0].phase, GateActuatorPhase::BeyondHorizon);
    let slow = walk(slow, 1.0, &[])?;
    assert_eq!(slow.states()?[0].arrival_count, 0);
    assert!(slow.outputs()?[0].position.opening_m > 0.0);
    assert!(slow.next_event()?.is_none());
    Ok(())
}
