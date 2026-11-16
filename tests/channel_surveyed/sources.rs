use super::support::{Reach, TestResult, check_water, forcing, near, profile, walls};
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelError, ChannelForcingInterval, ChannelGeometry,
    ChannelInitialState, ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput,
    ChannelReach, ChannelReachDefinition, ChannelSurveyedConfig, SurveyedChannelCell,
};
use hydr::forcing::WaterAmount;

pub fn definition(count: usize, roughness: f64, sources: bool) -> ChannelReachDefinition {
    let points = [[-5.0_f64, 4.0_f64], [-1.0, 0.0], [1.0, 0.0], [5.0, 4.0]];
    let profile = profile(
        "integer-trapezoid",
        &points.map(|p| p.map(f64::to_bits)),
        4.0_f64.to_bits(),
    );
    ChannelReachDefinition {
        vertical_datum: profile.vertical_datum.clone(),
        provenance: "Original integer trapezoid source acceptance".into(),
        geometry: ChannelGeometry::PrismaticSurveyed {
            profile,
            cells: (0..count)
                .map(|index| SurveyedChannelCell {
                    id: format!("cell-{index}"),
                    length_m: 4.0,
                    manning_n: roughness,
                })
                .collect(),
            numerical: ChannelSurveyedConfig::default(),
        },
        lateral_bindings: if sources {
            [
                ("in", ChannelLateralDirection::Inflow),
                ("out", ChannelLateralDirection::Withdrawal),
            ]
            .into_iter()
            .map(|(id, direction)| ChannelLateralBinding {
                boundary_id: id.into(),
                cell_id: "cell-0".into(),
                direction,
                ownership_assertion: format!("Independent surveyed source {id}"),
            })
            .collect()
        } else {
            Vec::new()
        },
        config: ChannelConfig::default(),
    }
}

pub fn initial(count: usize, moving: bool) -> ChannelInitialState {
    ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![12.0; count],
        longitudinal_momentum_m4_per_s: vec![if moving { 36.0 } else { 0.0 }; count],
    }
}

pub fn interval(
    start: f64,
    end: f64,
    incoming: WaterAmount,
    outgoing: WaterAmount,
) -> ChannelForcingInterval {
    let exterior = ChannelBoundary::Exterior {
        depth_m: 1.0,
        velocity_m_per_s: 3.0,
    };
    let mut result = forcing(start, end, [exterior.clone(), exterior]);
    result.lateral_inputs = vec![
        ChannelLateralInput::Inflow {
            boundary_id: "in".into(),
            amount: incoming,
            longitudinal_velocity_m_per_s: -1.0,
        },
        ChannelLateralInput::Withdrawal {
            boundary_id: "out".into(),
            amount: outgoing,
        },
    ];
    result
}

fn amount(value: f64, exact: bool) -> WaterAmount {
    if exact {
        WaterAmount::IntervalVolume { volume_m3: value }
    } else {
        WaterAmount::Rate {
            rate_m3_per_s: value / 0.01,
        }
    }
}

fn sourced(dry: bool, moving: bool) -> TestResult<Reach> {
    let mut state = initial(1, moving);
    if dry {
        state.volume_m3[0] = 0.0;
        state.longitudinal_momentum_m4_per_s[0] = 0.0;
    }
    Ok(ChannelReach::new(
        crate::runtime::client(),
        definition(1, 0.0, true),
        state,
    )?)
}

#[test]
fn rates_and_exact_volumes_use_original_donor_momentum() -> TestResult {
    for exact in [false, true] {
        let origin = sourced(false, true)?;
        let before = origin.snapshot_data()?;
        let held = interval(0.0, 0.01, amount(2.0, exact), amount(2.0, exact));
        let next = origin.step(0.01, &held)?.next;
        near(next.state()?[0].volume_m3, 12.0, 1e-11);
        near(next.state()?[0].longitudinal_momentum_m4_per_s, 28.0, 1e-10);
        let receipt = next.last_step()?.ok_or("Missing source receipt")?;
        assert_eq!(receipt.retries, 0);
        near(receipt.lateral_receipts[0].momentum_m4_per_s, -2.0, 1e-12);
        near(receipt.lateral_receipts[1].momentum_m4_per_s, -6.0, 1e-12);
        for (received, input) in receipt.lateral_receipts.iter().zip(&held.lateral_inputs) {
            assert_eq!(&received.input, input);
        }
        assert_eq!(origin.snapshot_data()?, before);
        check_water(&next)?;
    }
    Ok(())
}

#[test]
fn complete_drain_and_dry_donor_rejection_use_original_storage() -> TestResult {
    let origin = sourced(false, false)?;
    let mut held = interval(0.0, 0.01, amount(0.0, true), amount(12.0, true));
    held.left = ChannelBoundary::Wall;
    held.right = ChannelBoundary::Wall;
    let next = origin.step(0.01, &held)?.next;
    let state = next.state()?;
    assert!(state[0].dry);
    assert_eq!(state[0].volume_m3.to_bits(), 0.0_f64.to_bits());
    assert_eq!(state[0].longitudinal_momentum_m4_per_s, 0.0);
    check_water(&next)?;
    let dry = sourced(true, false)?;
    let before = dry.snapshot_data()?;
    held.lateral_inputs = interval(0.0, 0.01, amount(2.0, true), amount(1.0, true)).lateral_inputs;
    assert!(dry.step(0.01, &held).is_err());
    assert_eq!(dry.snapshot_data()?, before);
    Ok(())
}

#[test]
fn grouped_withdrawals_cannot_hide_original_excess_through_rounding() -> TestResult {
    let mut layout = definition(1, 0.0, true);
    layout.lateral_bindings[0].direction = ChannelLateralDirection::Withdrawal;
    let origin = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        layout,
        initial(1, false),
    )?;
    let first = 6.0;
    let second = f64::from_bits(6.0_f64.to_bits() + 1);
    assert_eq!(first + second, 12.0);
    assert!(first > 12.0 - second);
    let mut held = forcing(0.0, 0.01, walls());
    held.lateral_inputs = [("in", first), ("out", second)]
        .into_iter()
        .map(|(id, volume)| ChannelLateralInput::Withdrawal {
            boundary_id: id.into(),
            amount: amount(volume, true),
        })
        .collect();
    let before = origin.snapshot_data()?;
    assert!(origin.step(0.01, &held).is_err());
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

#[test]
fn face_export_and_withdrawal_share_original_donor_storage() -> TestResult {
    let origin = sourced(false, true)?;
    let before = origin.snapshot_data()?;
    let held = interval(0.0, 0.01, amount(0.0, false), amount(12.0, false));
    let next = origin.step(0.01, &held)?.next;
    assert!(next.time() > 0.0 && next.time() < 0.01);
    let receipt = next.last_step()?.ok_or("Missing shared donor receipt")?;
    let withdrawal = receipt.lateral_receipts[1].volume_m3;
    assert_eq!(withdrawal.to_bits(), (1200.0_f64 * next.time()).to_bits());
    let face_export = (-receipt.face_volume_m3[0]).max(0.0) + receipt.face_volume_m3[1].max(0.0);
    assert!(face_export > 0.0);
    assert!(face_export <= 12.0 - withdrawal);
    check_water(&next)?;
    let exact = interval(0.0, 0.01, amount(0.0, true), amount(12.0, true));
    assert!(matches!(
        origin.step(0.01, &exact),
        Err(ChannelError::RecomputeInterval { .. })
    ));
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

#[test]
fn source_candidate_cfl_retries_rates_and_requests_new_exact_integrals() -> TestResult {
    let origin = sourced(true, false)?;
    let before = origin.snapshot_data()?;
    let mut held = interval(
        0.0,
        1.0,
        WaterAmount::Rate { rate_m3_per_s: 2.0 },
        amount(0.0, false),
    );
    held.left = ChannelBoundary::Wall;
    held.right = ChannelBoundary::Wall;
    if let ChannelLateralInput::Inflow {
        longitudinal_velocity_m_per_s,
        ..
    } = &mut held.lateral_inputs[0]
    {
        *longitudinal_velocity_m_per_s = 20.0;
    }
    let next = origin.step(1.0, &held)?.next;
    assert!(next.time() > 0.0 && next.time() < 1.0);
    let receipt = next.last_step()?.ok_or("Missing candidate receipt")?;
    assert!(
        next.time()
            <= receipt
                .candidate_cfl_limit_s
                .ok_or("Missing candidate CFL")?
    );
    assert_eq!(receipt.accepted_start_cfl_limit_s, f64::INFINITY);
    near(receipt.lateral_in_m3, 2.0 * next.time(), 1e-12);
    check_water(&next)?;
    held.lateral_inputs[0] = ChannelLateralInput::Inflow {
        boundary_id: "in".into(),
        amount: WaterAmount::IntervalVolume { volume_m3: 2.0 },
        longitudinal_velocity_m_per_s: 20.0,
    };
    let error = origin
        .step(1.0, &held)
        .err()
        .ok_or("Exact source silently shortened")?;
    let suggested = match error {
        ChannelError::RecomputeInterval {
            suggested_end_s, ..
        } => suggested_end_s,
        other => return Err(format!("Unexpected source interval error: {other}").into()),
    };
    assert!(suggested > 0.0 && suggested < 1.0);
    held.end_s = suggested;
    let new_volume = 2.0 * suggested * suggested;
    held.lateral_inputs[0] = ChannelLateralInput::Inflow {
        boundary_id: "in".into(),
        amount: WaterAmount::IntervalVolume {
            volume_m3: new_volume,
        },
        longitudinal_velocity_m_per_s: 20.0,
    };
    let recomputed = origin.step(suggested, &held)?.next;
    near(recomputed.ledger().lateral_in_m3, new_volume, 1e-12);
    assert_eq!(origin.snapshot_data()?, before);
    check_water(&recomputed)
}

#[test]
fn explicit_zero_rates_preserve_native_arithmetic_and_partition() -> TestResult {
    let plain = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        definition(3, 0.03, false),
        initial(3, true),
    )?;
    let sourced = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        definition(3, 0.03, true),
        initial(3, true),
    )?;
    let held = interval(0.0, 0.001, amount(0.0, false), amount(0.0, false));
    let mut absent = held.clone();
    absent.lateral_inputs.clear();
    let a = plain.step(0.001, &absent)?.next;
    let b = sourced.step(0.001, &held)?.next;
    assert_eq!(a.state()?, b.state()?);
    let a = a.last_step()?.ok_or("Missing unforced receipt")?;
    let b = b.last_step()?.ok_or("Missing zero-source receipt")?;
    assert_eq!(a.end_s.to_bits(), b.end_s.to_bits());
    assert_eq!(a.face_volume_m3, b.face_volume_m3);
    assert_eq!(a.face_momentum_m4_per_s, b.face_momentum_m4_per_s);
    assert_eq!(a.friction_impulse_m4_per_s, b.friction_impulse_m4_per_s);
    assert_eq!(a.riemann, b.riemann);
    assert_eq!(a.friction_geometry, b.friction_geometry);
    assert_eq!(a.cfl_limit_s.to_bits(), b.cfl_limit_s.to_bits());
    assert_eq!(b.candidate_cfl_limit_s, None);
    Ok(())
}
