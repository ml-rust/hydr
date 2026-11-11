use super::fixtures::{
    binding, check_input_bits, exact, forcing, inflow, moving_cell, rate, withdrawal,
};
use crate::common::{TestResult, cell, check_water, definition, reach};
use hydr::channel::{ChannelError, ChannelLateralDirection};

#[test]
fn integrated_zero_and_positive_inputs_require_recomputation_before_shortening() -> TestResult {
    for volume in [0.0, -0.0, 2.0] {
        let mut layout = definition(vec![cell(0, 4.0, 2.0, 0.0)], 0.005);
        layout.lateral_bindings = vec![binding("in", "cell-0", ChannelLateralDirection::Inflow)];
        let initial = reach(layout, &[1.0], &[6.0])?;
        let before = initial.snapshot_data()?;
        let interval = forcing(0.0, 0.01, vec![inflow("in", exact(volume), -0.0)], true);
        let original = interval.clone();
        let error = initial
            .step(0.01, &interval)
            .err()
            .ok_or("integrated input silently shortened")?;
        let suggested_end = match error {
            ChannelError::RecomputeInterval {
                start_s,
                requested_end_s,
                suggested_end_s,
                ..
            } => {
                assert_eq!(start_s.to_bits(), 0.0_f64.to_bits());
                assert_eq!(requested_end_s.to_bits(), 0.01_f64.to_bits());
                assert!(suggested_end_s > 0.0 && suggested_end_s <= 0.005);
                suggested_end_s
            }
            other => return Err(format!("unexpected exact interval error: {other}").into()),
        };
        assert_eq!(initial.snapshot_data()?, before);
        check_input_bits(&interval.lateral_inputs[0], &original.lateral_inputs[0]);
        let recomputed = forcing(
            0.0,
            suggested_end,
            vec![inflow("in", exact(volume * suggested_end / 0.01), -0.0)],
            true,
        );
        let next = initial.step(suggested_end, &recomputed)?.next;
        assert_eq!(next.time().to_bits(), suggested_end.to_bits());
        assert_eq!(next.last_step()?.ok_or("missing exact receipt")?.retries, 0);
        check_water(&next)?;
    }
    Ok(())
}

#[test]
fn exact_inputs_require_the_original_forcing_span_bits() -> TestResult {
    let initial = moving_cell(vec![binding(
        "in",
        "cell-0",
        ChannelLateralDirection::Inflow,
    )])?;
    let before = initial.snapshot_data()?;
    let mut interval = forcing(0.0, 0.02, vec![inflow("in", exact(0.0), 0.0)], true);
    assert!(initial.step(0.01, &interval).is_err());
    interval.end_s = 0.01;
    interval.start_s = -0.0;
    assert!(initial.step(0.01, &interval).is_err());
    assert_eq!(initial.snapshot_data()?, before);
    let rate_interval = forcing(0.0, 0.02, vec![inflow("in", rate(0.0), 0.0)], true);
    assert_eq!(initial.step(0.01, &rate_interval)?.next.time(), 0.01);
    Ok(())
}

#[test]
fn explicit_zero_rates_preserve_unforced_state_and_step_partition() -> TestResult {
    let plain = reach(
        definition(vec![cell(0, 4.0, 2.0, 0.0)], 1.0),
        &[1.0],
        &[6.0],
    )?;
    let sourced = moving_cell(vec![
        binding("in", "cell-0", ChannelLateralDirection::Inflow),
        binding("out", "cell-0", ChannelLateralDirection::Withdrawal),
    ])?;
    let mut plain = plain;
    let mut sourced = sourced;
    for target in [0.01, 0.02, 0.03] {
        plain = plain
            .step(target, &forcing(plain.time(), target, vec![], true))?
            .next;
        sourced = sourced
            .step(
                target,
                &forcing(
                    sourced.time(),
                    target,
                    vec![inflow("in", rate(0.0), -0.0), withdrawal("out", rate(0.0))],
                    true,
                ),
            )?
            .next;
        assert_eq!(plain.time().to_bits(), sourced.time().to_bits());
        assert_eq!(plain.accepted_steps(), sourced.accepted_steps());
        assert_eq!(plain.state()?, sourced.state()?);
        let a = plain.last_step()?.ok_or("missing unforced receipt")?;
        let b = sourced.last_step()?.ok_or("missing zero-rate receipt")?;
        assert_eq!(a.retries, b.retries);
        assert_eq!(a.cfl_limit_s.to_bits(), b.cfl_limit_s.to_bits());
        assert_eq!(a.depletion_limit_s.to_bits(), b.depletion_limit_s.to_bits());
        assert_eq!(a.face_volume_m3, b.face_volume_m3);
        assert_eq!(a.face_momentum_m4_per_s, b.face_momentum_m4_per_s);
        assert_eq!(b.candidate_cfl_limit_s, None);
    }
    Ok(())
}
