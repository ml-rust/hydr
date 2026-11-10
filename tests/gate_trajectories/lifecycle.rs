use super::{fixture::*, reference_case};
use hydr::structures::*;

#[test]
fn immutable_evaluations_retain_policy_and_shuffled_queries_after_rejection() -> Result {
    let reference = case("binary_triangle")?;
    let mut limits = GateMotionLimits {
        maximum_opening_speed_m_per_s: 1.0,
        maximum_closing_speed_m_per_s: 1.0,
    };
    let mut accuracy = permissive();
    let definition = GateOpeningTrajectory::new(
        reference
            .knots
            .iter()
            .map(|&(time_s, opening_m)| GateOpeningKnot { time_s, opening_m })
            .collect(),
        limits,
        accuracy,
        "Immutable original policy".into(),
    )?;
    limits.maximum_opening_speed_m_per_s = 0.0;
    accuracy.maximum_absolute_opening_error_m = 0.0;
    assert_eq!(
        definition.motion_limits().maximum_opening_speed_m_per_s,
        1.0
    );
    assert_eq!(
        definition.accuracy().maximum_absolute_opening_error_m,
        f64::MAX
    );
    assert_eq!(limits.maximum_opening_speed_m_per_s, 0.0);
    assert_eq!(accuracy.maximum_absolute_opening_error_m, 0.0);
    let batch = {
        let source = reference_case::structure_batch(reference, 1)?;
        source.with_gate_trajectories(reference_case::assignments(&definition, 1))?
    };
    let accepted = batch.evaluate(0.5)?;
    let before = accepted.outputs()?;
    let operations = accepted.operations()?;
    for time in [4.0, 0.0, 3.0, 1.0, 0.5, 4.0, 0.5] {
        batch.evaluate(time)?;
    }
    expect_error(batch.evaluate(5.0), "UnsupportedRegime", None)?;
    assert_eq!(accepted.outputs()?, before);
    assert_eq!(accepted.operations()?, operations);
    assert_eq!(batch.evaluate(0.5)?.outputs()?, before);
    let rebuilt = reference_case::structure_batch(reference, 1)?
        .with_gate_trajectories(reference_case::assignments(&definition, 1))?;
    assert_eq!(rebuilt.evaluate(0.5)?.outputs()?, before);
    Ok(())
}

#[test]
fn held_proposals_keep_the_sampled_opening_and_do_not_advance_trajectory_state() -> Result {
    let batch = bind("binary_triangle", 1)?;
    let proposal = batch.propose_at(0.5, &[wet("gate-0")], &[])?;
    let operations = proposal.operations()?;
    let held = proposal.held_volumes(HeldHeadInterval {
        start: 0.0,
        end: 2.0,
    })?;
    let before = held.outputs()?;
    for output in &before {
        assert_eq!(output.proposal.operation, operations[0].operating_point);
        assert_eq!(
            output.volume_m3.to_bits(),
            (output.proposal.discharge_m3_per_s * 2.0).to_bits()
        );
    }
    batch.evaluate(4.0)?;
    expect_error(
        batch.propose_at(1.0, &[unknown("gate-0")], &[]),
        "UnknownSupport",
        Some(0),
    )?;
    assert_eq!(proposal.operations()?, operations);
    assert_eq!(held.outputs()?, before);
    Ok(())
}
