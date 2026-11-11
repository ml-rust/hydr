use super::common::{
    Reach, TestResult, cell, check_water, definition, exterior, forcing, near, reach, walls,
};
use hydr::channel::{
    ChannelError, ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput,
};
use hydr::forcing::WaterAmount;

fn sourced(max_step: f64, dry: bool) -> Result<Reach, Box<dyn std::error::Error>> {
    let mut layout = definition(vec![cell(0, 8.0)], max_step);
    layout.lateral_bindings = [
        ("in", ChannelLateralDirection::Inflow),
        ("out", ChannelLateralDirection::Withdrawal),
    ]
    .into_iter()
    .map(|(id, direction)| ChannelLateralBinding {
        boundary_id: id.into(),
        cell_id: "cell-0".into(),
        direction,
        ownership_assertion: format!("Independent synthetic source {id}"),
    })
    .collect();
    reach(
        layout,
        &[if dry { 0.0 } else { 1.0 }],
        &[if dry { 0.0 } else { 3.0 }],
    )
}

fn input(incoming: f64, outgoing: f64, exact: bool) -> Vec<ChannelLateralInput> {
    let amount = |volume_m3| {
        if exact {
            WaterAmount::IntervalVolume { volume_m3 }
        } else {
            WaterAmount::Rate {
                rate_m3_per_s: volume_m3 / 0.01,
            }
        }
    };
    vec![
        ChannelLateralInput::Inflow {
            boundary_id: "in".into(),
            amount: amount(incoming),
            longitudinal_velocity_m_per_s: -1.0,
        },
        ChannelLateralInput::Withdrawal {
            boundary_id: "out".into(),
            amount: amount(outgoing),
        },
    ]
}

#[test]
fn source_water_and_momentum_use_inflow_and_original_donor_velocities() -> TestResult {
    for exact in [false, true] {
        for (incoming, outgoing, volume, momentum) in [
            (2.0, 0.0, 10.0, 22.0),
            (0.0, 2.0, 6.0, 18.0),
            (2.0, 2.0, 8.0, 16.0),
        ] {
            let initial = sourced(1.0, false)?;
            let before = initial.snapshot_data()?;
            let mut interval = forcing(0.0, 0.01, [exterior(1.0, 3.0), exterior(1.0, 3.0)]);
            interval.lateral_inputs = input(incoming, outgoing, exact);
            let next = initial.step(0.01, &interval)?.next;
            near(next.state()?[0].volume_m3, volume, 1e-12);
            near(
                next.state()?[0].longitudinal_momentum_m4_per_s,
                momentum,
                1e-12,
            );
            let receipt = next.last_step()?.ok_or("Missing source receipt")?;
            assert_eq!(receipt.retries, 0);
            assert_eq!(
                receipt.lateral_receipts[0].input,
                interval.lateral_inputs[0]
            );
            assert_eq!(
                receipt.lateral_receipts[1].input,
                interval.lateral_inputs[1]
            );
            near(
                receipt.lateral_receipts[0].momentum_m4_per_s,
                -incoming,
                1e-12,
            );
            near(
                receipt.lateral_receipts[1].momentum_m4_per_s,
                -3.0 * outgoing,
                1e-12,
            );
            near(receipt.lateral_in_m3, incoming, 1e-12);
            near(receipt.lateral_out_m3, outgoing, 1e-12);
            assert_eq!(initial.snapshot_data()?, before);
            check_water(&next)?;
        }
    }
    Ok(())
}

#[test]
fn exact_source_recomputation_and_invalid_withdrawal_preserve_accepted_state() -> TestResult {
    let initial = sourced(0.005, false)?;
    let before = initial.snapshot_data()?;
    let mut interval = forcing(0.0, 0.01, [exterior(1.0, 3.0), exterior(1.0, 3.0)]);
    interval.lateral_inputs = input(0.002, 0.001, true);
    let error = initial
        .step(0.01, &interval)
        .err()
        .ok_or("Exact source silently shortened")?;
    let suggested = match error {
        ChannelError::RecomputeInterval {
            suggested_end_s, ..
        } => suggested_end_s,
        error => return Err(format!("Unexpected source interval error: {error}").into()),
    };
    assert!(suggested > 0.0 && suggested <= 0.005);
    assert_eq!(initial.snapshot_data()?, before);
    interval.end_s = suggested;
    // Recompute the supplied source law over the newly requested support.
    interval.lateral_inputs = input(0.2 * suggested, 0.1 * suggested, true);
    let next = initial.step(suggested, &interval)?.next;
    assert_eq!(next.time().to_bits(), suggested.to_bits());
    assert_eq!(
        next.last_step()?
            .ok_or("Missing recomputed receipt")?
            .retries,
        0
    );
    check_water(&next)?;

    let dry = sourced(1.0, true)?;
    let before = dry.snapshot_data()?;
    let mut interval = forcing(0.0, 0.01, walls());
    interval.lateral_inputs = input(2.0, 1.0, true);
    assert!(dry.step(0.01, &interval).is_err());
    assert_eq!(dry.snapshot_data()?, before);
    interval.lateral_inputs = input(2.0, 0.0, true);
    let wet = dry.step(0.01, &interval)?.next;
    near(wet.state()?[0].volume_m3, 2.0, 1e-12);
    near(wet.state()?[0].longitudinal_momentum_m4_per_s, -2.0, 1e-12);
    check_water(&wet)
}

fn continue_sources(
    mut current: Reach,
    steps: usize,
    exact: bool,
) -> Result<Reach, Box<dyn std::error::Error>> {
    for _ in 0..steps {
        let start = current.time();
        let end = start + 0.005;
        let mut interval = forcing(start, end, walls());
        interval.lateral_inputs = input(0.0001, 0.00005, exact);
        current = current.step(end, &interval)?.next;
        assert_eq!(current.time().to_bits(), end.to_bits());
        check_water(&current)?;
    }
    Ok(current)
}

#[test]
fn trusted_and_cold_restart_reproduce_complete_sourced_continuation() -> TestResult {
    for exact in [false, true] {
        let origin = sourced(1.0, false)?;
        let current = continue_sources(origin.clone(), 3, exact)?;
        let data = current.snapshot_data()?;
        let stored = data
            .last_interval
            .as_ref()
            .ok_or("Missing saved interval")?;
        let certificate = current
            .last_step()?
            .ok_or("Missing current interval")?
            .riemann
            .ok_or("Missing current Riemann certificate")?;
        assert_eq!(stored.diagnostics.riemann.as_ref(), Some(&certificate));
        let trusted = origin.restore(&current.snapshot())?;
        let cold = origin.import_snapshot(&data)?;
        assert_eq!(trusted.snapshot_data()?, data);
        assert_eq!(cold.snapshot_data()?, data);
        let expected = continue_sources(current, 3, exact)?.snapshot_data()?;
        assert_eq!(
            continue_sources(trusted, 3, exact)?.snapshot_data()?,
            expected
        );
        assert_eq!(continue_sources(cold, 3, exact)?.snapshot_data()?, expected);
        let mut altered = data.clone();
        altered
            .last_interval
            .as_mut()
            .ok_or("Missing restart interval")?
            .forcing
            .lateral_inputs[0] = ChannelLateralInput::Inflow {
            boundary_id: "in".into(),
            amount: WaterAmount::IntervalVolume { volume_m3: 0.3 },
            longitudinal_velocity_m_per_s: 1.0,
        };
        let before = origin.snapshot_data()?;
        assert!(origin.import_snapshot(&altered).is_err());
        assert_eq!(origin.snapshot_data()?, before);
        let mut altered_certificate = data.clone();
        altered_certificate
            .last_interval
            .as_mut()
            .ok_or("Missing saved interval")?
            .diagnostics
            .riemann
            .as_mut()
            .ok_or("Missing saved Riemann certificate")?
            .water_flux_width_m3_per_s[0] = f64::NAN;
        assert!(origin.import_snapshot(&altered_certificate).is_err());
        assert_eq!(origin.snapshot_data()?, before);
    }
    Ok(())
}
