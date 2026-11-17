//! Complete channel continuation and checkpoint admission.

#[path = "support/runtime.rs"]
mod runtime;

use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelGeometry, ChannelInitialState,
    ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput, ChannelReach,
    ChannelReachDefinition, RectangularChannelCell,
};
use hydr::forcing::WaterAmount;

#[path = "channel_restart/identity.rs"]
mod identity;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn reach() -> Result<ChannelReach<runtime::Runtime>, Box<dyn std::error::Error>> {
    Ok(ChannelReach::new(
        runtime::client(),
        ChannelReachDefinition {
            vertical_datum: "synthetic metres".into(),
            provenance: "rectangular restart fixture".into(),
            lateral_bindings: Vec::new(),
            geometry: ChannelGeometry::Rectangular {
                cells: (0..3)
                    .map(|i| RectangularChannelCell {
                        id: format!("cell-{i}"),
                        length_m: 2.0,
                        width_m: 2.0,
                        bed_elevation_m: 0.0,
                        maximum_depth_m: 4.0,
                        manning_n: 0.03,
                    })
                    .collect(),
            },
            config: ChannelConfig::default(),
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![4.0, 1.0, 2.0],
            longitudinal_momentum_m4_per_s: vec![0.8, -0.1, 0.0],
        },
    )?)
}

fn advance(
    mut reach: ChannelReach<runtime::Runtime>,
    steps: usize,
) -> Result<ChannelReach<runtime::Runtime>, Box<dyn std::error::Error>> {
    for _ in 0..steps {
        let time = reach.time();
        reach = reach
            .step(
                time + 0.02,
                &ChannelForcingInterval {
                    start_s: time,
                    end_s: time + 1.0,
                    left: ChannelBoundary::Wall,
                    right: ChannelBoundary::Wall,
                    lateral_inputs: Vec::new(),
                },
            )?
            .next;
    }
    Ok(reach)
}

#[test]
fn trusted_and_cold_restart_reproduce_identical_continuation() -> TestResult {
    let origin = reach()?;
    let current = advance(origin.clone(), 5)?;
    let trusted = origin.restore(&current.snapshot())?;
    let cold = origin.import_snapshot(&current.snapshot_data()?)?;
    assert_eq!(trusted.snapshot_data()?, current.snapshot_data()?);
    assert_eq!(cold.snapshot_data()?, current.snapshot_data()?);
    let expected = advance(current, 7)?.snapshot_data()?;
    assert_eq!(advance(trusted, 7)?.snapshot_data()?, expected);
    assert_eq!(advance(cold, 7)?.snapshot_data()?, expected);
    assert_eq!(origin.accepted_steps(), 0);
    Ok(())
}

#[test]
fn zero_step_import_requires_original_state_and_zero_counters() -> TestResult {
    let origin = reach()?;
    let data = origin.snapshot_data()?;
    assert_eq!(origin.import_snapshot(&data)?.snapshot_data()?, data);
    let mut altered = data.clone();
    altered.volume_m3[0] += 0.5;
    assert!(origin.import_snapshot(&altered).is_err());
    altered = data.clone();
    altered.ledger.boundary_in_m3 = 1.0;
    assert!(origin.import_snapshot(&altered).is_err());
    altered = data.clone();
    altered.time_s = 1.0;
    assert!(origin.import_snapshot(&altered).is_err());
    altered = data.clone();
    altered.ledger.numerical_transfer_underflow_bound_m3 = f64::from_bits(1);
    assert!(origin.import_snapshot(&altered).is_err());
    Ok(())
}

#[test]
fn cold_import_rejects_altered_bounded_continuation_fields() -> TestResult {
    let origin = reach()?;
    let current = advance(origin.clone(), 3)?;
    let data = current.snapshot_data()?;
    let unchanged = origin.snapshot_data()?;
    for mutation in 0..29 {
        let mut altered = data.clone();
        let last = altered
            .last_interval
            .as_mut()
            .ok_or("missing fixture interval")?;
        match mutation {
            0 => altered.volume_m3[0] = f64::from_bits(altered.volume_m3[0].to_bits() + 1),
            1 => altered.longitudinal_momentum_m4_per_s[0] += 1.0,
            2 => altered.time_s += 0.001,
            3 => altered.accepted_steps += 1,
            4 => last.previous_volume_m3[0] = f64::NAN,
            5 => last.previous_ledger.boundary_out_m3 = -1.0,
            6 => last.previous_time_s = altered.time_s,
            7 => last.forcing.start_s += 0.001,
            8 => last.requested_end_s = last.previous_time_s,
            9 => last.diagnostics.face_volume_m3[1] += 0.001,
            10 => last.diagnostics.cfl_limit_s *= 0.5,
            11 => {
                last.previous_accepted_steps = usize::MAX;
                altered.accepted_steps = usize::MAX;
            }
            12 => {
                last.diagnostics.face_volume_m3.pop();
            }
            13 => last.diagnostics.friction_impulse_m4_per_s[0] = f64::NAN,
            14 => altered.ledger.initial_storage_m3 += 1.0,
            15 => last.diagnostics.max_local_momentum_residual_m4_per_s = f64::NAN,
            16 => last.diagnostics.max_momentum_underflow_bound_m4_per_s = f64::NAN,
            17 => {
                last.diagnostics.momentum_underflow_bound_m4_per_s.pop();
            }
            18 => last.diagnostics.momentum_underflow_bound_m4_per_s[0] = f64::from_bits(1),
            19 => altered.ledger.numerical_transfer_underflow_bound_m3 = -1.0,
            20 => altered.ledger.numerical_transfer_underflow_bound_m3 = f64::NAN,
            21 => {
                last.previous_ledger.numerical_transfer_underflow_bound_m3 = 1.0;
            }
            22 => last.diagnostics.transfer_underflow_bound_m3 = f64::NAN,
            23 => {
                last.diagnostics.face_transfer_underflow_bound_m3.pop();
            }
            24 => last.diagnostics.face_transfer_underflow_bound_m3[0] = -1.0,
            25 => last.diagnostics.face_transfer_underflow_bound_m3[0] = f64::from_bits(1),
            26 => last.diagnostics.face_transfer_underflow_bound_m3[0] = f64::NAN,
            27 => {
                altered.ledger.numerical_transfer_underflow_bound_m3 = f64::from_bits(1);
            }
            28 => last.previous_ledger.numerical_transfer_underflow_bound_m3 = -1.0,
            _ => return Err("unrecognized fixture mutation".into()),
        }
        assert!(
            origin.import_snapshot(&altered).is_err(),
            "mutation {mutation}"
        );
        assert_eq!(origin.snapshot_data()?, unchanged);
    }
    Ok(())
}

#[test]
fn snapshot_identity_preserves_signed_zero_geometry_and_epoch() -> TestResult {
    let origin = reach()?;
    let mut altered = origin.snapshot_data()?;
    let ChannelGeometry::Rectangular { cells } = &mut altered.definition.geometry else {
        unreachable!()
    };
    cells[0].bed_elevation_m = -0.0;
    assert!(origin.import_snapshot(&altered).is_err());
    altered = origin.snapshot_data()?;
    altered.initial.time_s = -0.0;
    assert!(origin.import_snapshot(&altered).is_err());
    altered = origin.snapshot_data()?;
    altered.initial.longitudinal_momentum_m4_per_s[2] = -0.0;
    assert!(origin.import_snapshot(&altered).is_err());
    Ok(())
}

fn sourced_reach() -> Result<ChannelReach<runtime::Runtime>, Box<dyn std::error::Error>> {
    Ok(ChannelReach::new(
        runtime::client(),
        ChannelReachDefinition {
            vertical_datum: "synthetic metres".into(),
            provenance: "rectangular lateral restart fixture".into(),
            geometry: ChannelGeometry::Rectangular {
                cells: vec![RectangularChannelCell {
                    id: "cell".into(),
                    length_m: 4.0,
                    width_m: 2.0,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 10.0,
                    manning_n: 0.0,
                }],
            },
            lateral_bindings: [
                ("inlet", ChannelLateralDirection::Inflow),
                ("outlet", ChannelLateralDirection::Withdrawal),
            ]
            .into_iter()
            .map(|(id, direction)| ChannelLateralBinding {
                boundary_id: id.into(),
                cell_id: "cell".into(),
                direction,
                ownership_assertion: "separate explicitly selected restart inputs".into(),
            })
            .collect(),
            config: ChannelConfig::default(),
        },
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![8.0],
            longitudinal_momentum_m4_per_s: vec![24.0],
        },
    )?)
}

fn sourced_advance(
    mut reach: ChannelReach<runtime::Runtime>,
    kind: usize,
    steps: usize,
) -> Result<ChannelReach<runtime::Runtime>, Box<dyn std::error::Error>> {
    for _ in 0..steps {
        let time = reach.time();
        let end = time + 0.005;
        let state = reach
            .state()?
            .into_iter()
            .next()
            .ok_or("missing fixture cell")?;
        let boundary = ChannelBoundary::Exterior {
            depth_m: state.depth_m,
            velocity_m_per_s: state.velocity_m_per_s.ok_or("dry fixture cell")?,
        };
        let incoming = if kind == 0 {
            WaterAmount::Rate { rate_m3_per_s: 0.2 }
        } else {
            WaterAmount::IntervalVolume { volume_m3: 0.001 }
        };
        let outgoing = if kind == 1 {
            WaterAmount::IntervalVolume { volume_m3: 0.0005 }
        } else {
            WaterAmount::Rate { rate_m3_per_s: 0.1 }
        };
        reach = reach
            .step(
                end,
                &ChannelForcingInterval {
                    start_s: time,
                    end_s: end,
                    left: boundary.clone(),
                    right: boundary,
                    lateral_inputs: vec![
                        ChannelLateralInput::Withdrawal {
                            boundary_id: "outlet".into(),
                            amount: outgoing,
                        },
                        ChannelLateralInput::Inflow {
                            boundary_id: "inlet".into(),
                            amount: incoming,
                            longitudinal_velocity_m_per_s: -1.0,
                        },
                    ],
                },
            )?
            .next;
    }
    Ok(reach)
}

#[test]
fn rate_exact_and_mixed_sources_resume_with_original_receipts() -> TestResult {
    for kind in 0..3 {
        let origin = sourced_reach()?;
        let current = sourced_advance(origin.clone(), kind, 3)?;
        let data = current.snapshot_data()?;
        let cold = origin.import_snapshot(&data)?;
        let trusted = origin.restore(&current.snapshot())?;
        assert_eq!(cold.snapshot_data()?, data);
        assert_eq!(trusted.snapshot_data()?, data);
        let expected = sourced_advance(current, kind, 3)?.snapshot_data()?;
        assert_eq!(sourced_advance(cold, kind, 3)?.snapshot_data()?, expected);
        assert_eq!(
            sourced_advance(trusted, kind, 3)?.snapshot_data()?,
            expected
        );
    }
    Ok(())
}

#[test]
fn lateral_import_rejects_changed_sources_counters_witnesses_and_bounds() -> TestResult {
    let origin = sourced_reach()?;
    let data = sourced_advance(origin.clone(), 2, 3)?.snapshot_data()?;
    let unchanged = origin.snapshot_data()?;
    for mutation in 0..23 {
        let mut altered = data.clone();
        let last = altered
            .last_interval
            .as_mut()
            .ok_or("missing source interval")?;
        let receipt = last
            .diagnostics
            .lateral_receipts
            .first_mut()
            .ok_or("missing source receipt")?;
        match mutation {
            0 => altered.definition.lateral_bindings[0].boundary_id = "other".into(),
            1 => last.forcing.lateral_inputs.clear(),
            2 => {
                last.forcing.lateral_inputs[0] = ChannelLateralInput::Withdrawal {
                    boundary_id: "outlet".into(),
                    amount: WaterAmount::IntervalVolume { volume_m3: 0.0005 },
                }
            }
            3 => {
                last.forcing.lateral_inputs[1] = ChannelLateralInput::Inflow {
                    boundary_id: "inlet".into(),
                    amount: WaterAmount::IntervalVolume { volume_m3: 0.002 },
                    longitudinal_velocity_m_per_s: -1.0,
                }
            }
            4 => {
                last.forcing.lateral_inputs[1] = ChannelLateralInput::Inflow {
                    boundary_id: "inlet".into(),
                    amount: WaterAmount::IntervalVolume { volume_m3: 0.001 },
                    longitudinal_velocity_m_per_s: 1.0,
                }
            }
            5 => altered.lateral.cumulative_volume_m3[0] += 1.0,
            6 => altered.lateral.counter_roundoff_bound_m3[0] = f64::NAN,
            7 => altered.lateral.cumulative_assembly_bound_m3 = -1.0,
            8 => last.previous_lateral.cumulative_volume_m3[0] += 1.0,
            9 => last.previous_lateral.counter_roundoff_bound_m3[0] = -1.0,
            10 => receipt.counter_addition_low_m3 = f64::from_bits(1),
            11 => receipt.counter_bound_after_m3 += 0.01,
            12 => receipt.applied_longitudinal_velocity_m_per_s = None,
            13 => {
                receipt.input = ChannelLateralInput::Inflow {
                    boundary_id: "other".into(),
                    amount: WaterAmount::Rate { rate_m3_per_s: 0.2 },
                    longitudinal_velocity_m_per_s: -1.0,
                }
            }
            14 => last.diagnostics.candidate_cfl_limit_s = None,
            15 => last.diagnostics.lateral_momentum_abs_bound_m4_per_s.clear(),
            16 => last.diagnostics.lateral_assembly_abs_bound_m3 = f64::NAN,
            17 => last.diagnostics.lateral_counter_addition_abs_bound_m3 = -1.0,
            18 => altered.ledger.lateral_in_m3 += 0.1,
            19 => altered.ledger.lateral_counter_abs_bound_m3 = f64::NAN,
            20 => last.previous_ledger.lateral_assembly_abs_bound_m3 += 1.0,
            21 => last.forcing.end_s += 0.001,
            22 => {
                last.previous_lateral.cumulative_volume_m3.pop();
            }
            _ => return Err("unrecognized source mutation".into()),
        }
        assert!(
            origin.import_snapshot(&altered).is_err(),
            "source mutation {mutation}"
        );
        assert_eq!(origin.snapshot_data()?, unchanged);
    }
    Ok(())
}

#[test]
fn exact_zero_sources_preserve_span_and_zero_history_requirements() -> TestResult {
    let origin = sourced_reach()?;
    let unchanged = origin.snapshot_data()?;
    let forcing = ChannelForcingInterval {
        start_s: 0.0,
        end_s: 0.005,
        left: ChannelBoundary::Wall,
        right: ChannelBoundary::Wall,
        lateral_inputs: vec![
            ChannelLateralInput::Inflow {
                boundary_id: "inlet".into(),
                amount: WaterAmount::IntervalVolume { volume_m3: 0.0 },
                longitudinal_velocity_m_per_s: -0.0,
            },
            ChannelLateralInput::Withdrawal {
                boundary_id: "outlet".into(),
                amount: WaterAmount::Rate { rate_m3_per_s: 0.0 },
            },
        ],
    };
    let accepted = origin.step(0.005, &forcing)?.next;
    let data = accepted.snapshot_data()?;
    let stored_input = data
        .last_interval
        .as_ref()
        .ok_or("missing zero interval")?
        .forcing
        .lateral_inputs
        .first()
        .ok_or("missing zero input")?;
    assert!(matches!(stored_input, ChannelLateralInput::Inflow {
        amount: WaterAmount::IntervalVolume { volume_m3 }, longitudinal_velocity_m_per_s, ..
    } if volume_m3.to_bits() == 0.0f64.to_bits()
        && longitudinal_velocity_m_per_s.to_bits() == (-0.0f64).to_bits()));
    assert_eq!(origin.import_snapshot(&data)?.snapshot_data()?, data);
    let mut altered = data.clone();
    altered
        .last_interval
        .as_mut()
        .ok_or("missing zero interval")?
        .forcing
        .end_s = 0.006;
    assert!(origin.import_snapshot(&altered).is_err());
    let mut altered_zero = unchanged.clone();
    altered_zero.lateral.cumulative_volume_m3[0] = f64::from_bits(1);
    assert!(origin.import_snapshot(&altered_zero).is_err());
    assert_eq!(origin.snapshot_data()?, unchanged);
    Ok(())
}
