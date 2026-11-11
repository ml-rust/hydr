use super::fixtures::{
    binding, check_input_bits, exact, forcing, inflow, input_id, near, rate, withdrawal,
};
use crate::common::{TestResult, cell, check_water, definition, exterior, reach};
use hydr::channel::{ChannelError, ChannelLateralDirection};

#[test]
fn multiple_inputs_share_a_cell_without_losing_identity_or_signed_momentum() -> TestResult {
    for (first_volume, first_velocity, second_volume, second_velocity, expected_momentum) in
        [(0.75, 2.0, 1.25, -2.0, -1.0), (1.0, 2.0, 1.0, -2.0, 0.0)]
    {
        let mut layout = definition(vec![cell(0, 4.0, 2.0, 0.0)], 1.0);
        layout.lateral_bindings = vec![
            binding("first", "cell-0", ChannelLateralDirection::Inflow),
            binding("second", "cell-0", ChannelLateralDirection::Inflow),
        ];
        let initial = reach(layout, &[1.0], &[0.0])?;
        let inputs = vec![
            inflow("first", exact(first_volume), first_velocity),
            inflow("second", exact(second_volume), second_velocity),
        ];
        let mut reversed = inputs.clone();
        reversed.reverse();
        let first = initial
            .step(0.01, &forcing(0.0, 0.01, inputs.clone(), false))?
            .next;
        let second = initial
            .step(0.01, &forcing(0.0, 0.01, reversed.clone(), false))?
            .next;
        assert_eq!(first.state()?, second.state()?);
        assert_eq!(first.lateral_ledgers()?, second.lateral_ledgers()?);
        near(first.state()?[0].volume_m3, 10.0);
        near(
            first.state()?[0].longitudinal_momentum_m4_per_s,
            expected_momentum,
        );
        for result in [&first, &second] {
            let receipt = result.last_step()?.ok_or("missing grouped receipt")?;
            for input in &inputs {
                let retained = receipt
                    .lateral_receipts
                    .iter()
                    .find(|value| input_id(&value.input) == input_id(input))
                    .ok_or("missing grouped input")?;
                check_input_bits(&retained.input, input);
            }
            check_water(result)?;
        }
        let data = second.snapshot_data()?;
        let stored = &data
            .last_interval
            .as_ref()
            .ok_or("missing bounded forcing")?
            .forcing
            .lateral_inputs;
        assert_eq!(stored.len(), reversed.len());
        for (actual, expected) in stored.iter().zip(&reversed) {
            check_input_bits(actual, expected);
        }
    }
    Ok(())
}

#[test]
fn positive_external_water_cannot_disappear_in_a_large_storage_addition() -> TestResult {
    let mut layout = definition(vec![cell(0, 1.0, 2.0_f64.powi(53), 0.0)], 1.0);
    layout.lateral_bindings = vec![binding("in", "cell-0", ChannelLateralDirection::Inflow)];
    let initial = reach(layout, &[1.0], &[0.0])?;
    let before = initial.snapshot_data()?;
    assert!(
        initial
            .step(
                0.001,
                &forcing(0.0, 0.001, vec![inflow("in", exact(1.0), 0.0)], false)
            )
            .is_err()
    );
    assert_eq!(initial.snapshot_data()?, before);
    Ok(())
}

#[test]
fn smallest_exact_inflow_retains_its_original_operand_and_positive_storage() -> TestResult {
    let delta = f64::from_bits(1);
    let mut layout = definition(vec![cell(0, 1.0, 1.0, 0.0)], 1.0);
    layout.lateral_bindings = vec![binding("in", "cell-0", ChannelLateralDirection::Inflow)];
    let initial = reach(layout, &[delta], &[0.0])?;
    let input = inflow("in", exact(delta), 0.0);
    let next = initial
        .step(0.001, &forcing(0.0, 0.001, vec![input.clone()], false))?
        .next;
    assert_eq!(next.time(), 0.001);
    assert_eq!(
        next.state()?[0].volume_m3.to_bits(),
        f64::from_bits(2).to_bits()
    );
    assert!(!next.state()?[0].dry);
    let receipt = next.last_step()?.ok_or("missing smallest-input receipt")?;
    assert_eq!(receipt.retries, 0);
    assert_eq!(
        receipt.lateral_receipts[0].volume_m3.to_bits(),
        delta.to_bits()
    );
    check_input_bits(&receipt.lateral_receipts[0].input, &input);
    check_water(&next)
}

#[test]
fn grouped_exact_withdrawals_reject_original_volume_exceeding_the_donor() -> TestResult {
    let first_volume = 4.0;
    let second_volume = f64::from_bits(4.0_f64.to_bits() + 1);
    // The rounded sum hides an original excess of 2^-50 m³.
    assert_eq!(first_volume + second_volume, 8.0);
    assert!(first_volume > 8.0 - second_volume);
    let mut layout = definition(vec![cell(0, 4.0, 2.0, 0.0)], 1.0);
    layout.lateral_bindings = vec![
        binding("first", "cell-0", ChannelLateralDirection::Withdrawal),
        binding("second", "cell-0", ChannelLateralDirection::Withdrawal),
    ];
    let initial = reach(layout, &[1.0], &[0.0])?;
    let before = initial.snapshot_data()?;
    let interval = forcing(
        0.0,
        0.01,
        vec![
            withdrawal("first", exact(first_volume)),
            withdrawal("second", exact(second_volume)),
        ],
        false,
    );
    let error = initial
        .step(0.01, &interval)
        .err()
        .ok_or("grouped exact withdrawals overdraw the original donor")?;
    assert!(matches!(error, ChannelError::RecomputeInterval {
        start_s, requested_end_s, suggested_end_s, ..
    } if start_s == 0.0 && requested_end_s == 0.01 && suggested_end_s > 0.0 && suggested_end_s < 0.01));
    assert_eq!(initial.snapshot_data()?, before);
    Ok(())
}

#[test]
fn grouped_withdrawal_rates_shorten_before_overdrawing_the_original_donor() -> TestResult {
    let first_rate = 4.0 / 0.01;
    let second_rate = f64::from_bits(4.0_f64.to_bits() + 1) / 0.01;
    let mut layout = definition(vec![cell(0, 4.0, 2.0, 0.0)], 1.0);
    layout.lateral_bindings = vec![
        binding("first", "cell-0", ChannelLateralDirection::Withdrawal),
        binding("second", "cell-0", ChannelLateralDirection::Withdrawal),
    ];
    let initial = reach(layout, &[1.0], &[0.0])?;
    let before = initial.snapshot_data()?;
    let next = initial
        .step(
            0.01,
            &forcing(
                0.0,
                0.01,
                vec![
                    withdrawal("first", rate(first_rate)),
                    withdrawal("second", rate(second_rate)),
                ],
                false,
            ),
        )?
        .next;
    assert!(next.time() > 0.0 && next.time() < 0.01);
    let receipt = next
        .last_step()?
        .ok_or("missing grouped withdrawal receipt")?;
    let first = receipt
        .lateral_receipts
        .iter()
        .find(|value| input_id(&value.input) == "first")
        .ok_or("missing first withdrawal")?
        .volume_m3;
    let second = receipt
        .lateral_receipts
        .iter()
        .find(|value| input_id(&value.input) == "second")
        .ok_or("missing second withdrawal")?
        .volume_m3;
    assert_eq!(first.to_bits(), (first_rate * next.time()).to_bits());
    assert_eq!(second.to_bits(), (second_rate * next.time()).to_bits());
    assert!(first <= 8.0 - second);
    near(next.state()?[0].volume_m3, (8.0 - first) - second);
    assert_eq!(initial.snapshot_data()?, before);
    check_water(&next)
}

#[test]
fn exact_withdrawal_and_face_export_cannot_use_matching_incoming_water() -> TestResult {
    let requested_end = 0.035;
    let withdrawal_volume = 7.125;
    assert_eq!((8.0 - withdrawal_volume) / 25.0, requested_end);
    assert!(25.0 * requested_end > 8.0 - withdrawal_volume);
    let mut layout = definition(vec![cell(0, 4.0, 2.0, 0.0)], 1.0);
    layout.lateral_bindings = vec![binding(
        "out",
        "cell-0",
        ChannelLateralDirection::Withdrawal,
    )];
    let initial = reach(layout, &[1.0], &[25.0])?;
    assert_eq!(initial.state()?[0].volume_m3, 8.0);
    assert_eq!(initial.state()?[0].longitudinal_momentum_m4_per_s, 100.0);
    let before = initial.snapshot_data()?;
    let mut interval = forcing(
        0.0,
        requested_end,
        vec![withdrawal("out", exact(withdrawal_volume))],
        true,
    );
    interval.left = exterior(1.0, 12.5);
    interval.right = exterior(1.0, 12.5);
    let error = initial
        .step(requested_end, &interval)
        .err()
        .ok_or("incoming face water hides an original-donor overdraw")?;
    assert!(matches!(error, ChannelError::RecomputeInterval {
        start_s, requested_end_s, suggested_end_s, ..
    } if start_s == 0.0 && requested_end_s == requested_end && suggested_end_s > 0.0 && suggested_end_s < requested_end));
    assert_eq!(initial.snapshot_data()?, before);
    Ok(())
}

#[test]
fn withdrawal_rate_and_face_export_shorten_before_using_incoming_water() -> TestResult {
    let requested_end = 0.035;
    let withdrawal_rate = 7.125 / requested_end;
    let mut layout = definition(vec![cell(0, 4.0, 2.0, 0.0)], 1.0);
    layout.lateral_bindings = vec![binding(
        "out",
        "cell-0",
        ChannelLateralDirection::Withdrawal,
    )];
    let initial = reach(layout, &[1.0], &[25.0])?;
    let before = initial.snapshot_data()?;
    let mut interval = forcing(
        0.0,
        requested_end,
        vec![withdrawal("out", rate(withdrawal_rate))],
        true,
    );
    interval.left = exterior(1.0, 12.5);
    interval.right = exterior(1.0, 12.5);
    let next = initial.step(requested_end, &interval)?.next;
    assert!(next.time() > 0.0 && next.time() < requested_end);
    let receipt = next
        .last_step()?
        .ok_or("missing withdrawal and face-export receipt")?;
    assert_eq!(receipt.face_volume_m3.len(), 2);
    assert!(receipt.face_volume_m3[0] > 0.0);
    assert!(receipt.face_volume_m3[1] > 0.0);
    let withdrawal = receipt.lateral_receipts[0].volume_m3;
    assert_eq!(
        withdrawal.to_bits(),
        (withdrawal_rate * next.time()).to_bits()
    );
    assert_eq!(
        receipt.face_volume_m3[1].to_bits(),
        (25.0 * next.time()).to_bits()
    );
    let outgoing = receipt.face_volume_m3[1].max(0.0) + (-receipt.face_volume_m3[0]).max(0.0);
    assert!(outgoing <= 8.0 - withdrawal);
    assert_eq!(initial.snapshot_data()?, before);
    check_water(&next)
}
