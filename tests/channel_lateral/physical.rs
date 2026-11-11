use super::fixtures::{
    binding, check_input_bits, exact, forcing, inflow, input_id, moving_cell, near, rate,
    withdrawal,
};
use crate::common::{TestResult, cell, check_water, definition, reach};
use hydr::channel::{ChannelError, ChannelLateralDirection};

#[test]
fn source_volumes_and_donor_momentum_match_the_declared_inputs() -> TestResult {
    for (input_amount, withdrawal_amount) in [
        (exact(2.0), exact(2.0)),
        (rate(200.0), rate(200.0)),
        (exact(2.0), rate(200.0)),
    ] {
        for (has_inflow, has_withdrawal, expected_volume, expected_momentum) in [
            (true, false, 10.0, 22.0),
            (false, true, 6.0, 18.0),
            (true, true, 8.0, 16.0),
        ] {
            let mut bindings = Vec::new();
            let mut inputs = Vec::new();
            if has_inflow {
                bindings.push(binding("in", "cell-0", ChannelLateralDirection::Inflow));
                inputs.push(inflow("in", input_amount, -1.0));
            }
            if has_withdrawal {
                bindings.push(binding(
                    "out",
                    "cell-0",
                    ChannelLateralDirection::Withdrawal,
                ));
                inputs.push(withdrawal("out", withdrawal_amount));
            }
            let initial = moving_cell(bindings)?;
            let interval = forcing(0.0, 0.01, inputs.clone(), true);
            let next = initial.step(0.01, &interval)?.next;
            assert_eq!(next.time(), 0.01);
            let state = next.state()?;
            near(state[0].volume_m3, expected_volume);
            near(state[0].longitudinal_momentum_m4_per_s, expected_momentum);
            if !has_inflow {
                near(
                    state[0].velocity_m_per_s.ok_or("missing wet velocity")?,
                    3.0,
                );
            }
            let receipt = next.last_step()?.ok_or("missing source receipt")?;
            assert_eq!(receipt.retries, 0);
            near(receipt.lateral_in_m3, if has_inflow { 2.0 } else { 0.0 });
            near(
                receipt.lateral_out_m3,
                if has_withdrawal { 2.0 } else { 0.0 },
            );
            for input in &inputs {
                let received = receipt
                    .lateral_receipts
                    .iter()
                    .find(|value| input_id(&value.input) == input_id(input))
                    .ok_or("missing binding receipt")?;
                check_input_bits(&received.input, input);
                near(received.volume_m3, 2.0);
                near(
                    received.momentum_m4_per_s,
                    if input_id(input) == "in" { -2.0 } else { -6.0 },
                );
                assert_eq!(
                    received.applied_longitudinal_velocity_m_per_s,
                    Some(if input_id(input) == "in" { -1.0 } else { 3.0 })
                );
            }
            let checkpoint = next.snapshot_data()?;
            let retained = &checkpoint
                .last_interval
                .as_ref()
                .ok_or("missing bounded interval")?
                .forcing;
            for (actual, expected) in retained.lateral_inputs.iter().zip(&inputs) {
                check_input_bits(actual, expected);
            }
            check_water(&next)?;
        }
    }
    Ok(())
}

#[test]
fn dry_receivers_accept_prescribed_momentum_without_inventing_a_donor_velocity() -> TestResult {
    let mut layout = definition(vec![cell(0, 4.0, 2.0, 0.0)], 1.0);
    layout.lateral_bindings = vec![binding("in", "cell-0", ChannelLateralDirection::Inflow)];
    let initial = reach(layout, &[0.0], &[0.0])?;
    let next = initial
        .step(
            0.01,
            &forcing(0.0, 0.01, vec![inflow("in", exact(2.0), -1.0)], false),
        )?
        .next;
    near(next.state()?[0].volume_m3, 2.0);
    near(next.state()?[0].longitudinal_momentum_m4_per_s, -2.0);
    check_water(&next)
}

#[test]
fn incoming_water_cannot_fund_an_originally_dry_withdrawal() -> TestResult {
    let mut layout = definition(vec![cell(0, 4.0, 2.0, 0.0)], 1.0);
    layout.lateral_bindings = vec![
        binding("in", "cell-0", ChannelLateralDirection::Inflow),
        binding("out", "cell-0", ChannelLateralDirection::Withdrawal),
    ];
    let initial = reach(layout, &[0.0], &[0.0])?;
    let before = initial.snapshot_data()?;
    let result = initial.step(
        0.01,
        &forcing(
            0.0,
            0.01,
            vec![inflow("in", exact(2.0), 0.0), withdrawal("out", exact(1.0))],
            false,
        ),
    );
    assert!(result.is_err());
    assert_eq!(initial.snapshot_data()?, before);
    Ok(())
}

#[test]
fn combined_face_export_and_withdrawal_limit_the_original_donor() -> TestResult {
    let initial = moving_cell(vec![binding(
        "out",
        "cell-0",
        ChannelLateralDirection::Withdrawal,
    )])?;
    let before = initial.snapshot_data()?;
    let next = initial
        .step(
            0.01,
            &forcing(0.0, 0.01, vec![withdrawal("out", rate(800.0))], true),
        )?
        .next;
    assert!(next.time() > 0.0 && next.time() < 0.01);
    let receipt = next.last_step()?.ok_or("missing depletion receipt")?;
    assert!(next.time() <= receipt.depletion_limit_s);
    assert!(receipt.lateral_out_m3 + receipt.face_volume_m3[1].max(0.0) <= 8.0);
    assert_eq!(initial.snapshot_data()?, before);
    check_water(&next)
}

#[test]
fn complete_still_withdrawal_preserves_exact_dry_state() -> TestResult {
    let mut layout = definition(vec![cell(0, 4.0, 2.0, 0.0)], 1.0);
    layout.lateral_bindings = vec![binding(
        "out",
        "cell-0",
        ChannelLateralDirection::Withdrawal,
    )];
    let initial = reach(layout, &[1.0], &[0.0])?;
    let next = initial
        .step(
            0.01,
            &forcing(0.0, 0.01, vec![withdrawal("out", exact(8.0))], false),
        )?
        .next;
    let state = next.state()?;
    assert!(state[0].dry);
    assert_eq!(state[0].volume_m3, 0.0);
    assert_eq!(state[0].longitudinal_momentum_m4_per_s, 0.0);
    assert_eq!(next.time(), 0.01);
    assert_eq!(next.last_step()?.ok_or("missing drain receipt")?.retries, 0);
    check_water(&next)
}

#[test]
fn strong_dry_injection_obeys_candidate_wave_speeds() -> TestResult {
    let mut layout = definition(vec![cell(0, 1.0, 2.0, 0.0)], 1.0);
    layout.lateral_bindings = vec![binding("in", "cell-0", ChannelLateralDirection::Inflow)];
    let initial = reach(layout, &[0.0], &[0.0])?;
    let before = initial.snapshot_data()?;
    let next = initial
        .step(
            1.0,
            &forcing(0.0, 1.0, vec![inflow("in", rate(2.0), 20.0)], false),
        )?
        .next;
    assert!(next.time() > 0.0 && next.time() < 0.02);
    let receipt = next.last_step()?.ok_or("missing injection receipt")?;
    assert_eq!(receipt.accepted_start_cfl_limit_s, f64::INFINITY);
    let candidate_limit = receipt
        .candidate_cfl_limit_s
        .ok_or("missing candidate CFL")?;
    assert!(candidate_limit.is_finite() && next.time() <= candidate_limit);
    check_water(&next)?;
    let error = initial
        .step(
            1.0,
            &forcing(0.0, 1.0, vec![inflow("in", exact(2.0), 20.0)], false),
        )
        .err()
        .ok_or("exact injection accepted original long interval")?;
    assert!(
        matches!(error, ChannelError::RecomputeInterval { suggested_end_s, .. } if suggested_end_s > 0.0 && suggested_end_s < 1.0)
    );
    assert_eq!(initial.snapshot_data()?, before);
    Ok(())
}
