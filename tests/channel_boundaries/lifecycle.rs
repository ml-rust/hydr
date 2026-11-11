use super::common::{
    Reach, Shape, Side, TestResult, check_water, definition, initial, interval, reach,
};
use hydr::channel::{
    ChannelBoundary, ChannelError, ChannelForcingInterval, ChannelLateralBinding,
    ChannelLateralDirection, ChannelLateralInput,
};
use hydr::forcing::WaterAmount;

fn source_layout(
    shape: Shape,
    max_step: f64,
    direction: ChannelLateralDirection,
) -> hydr::channel::ChannelReachDefinition {
    let mut layout = definition(shape, 1, 1.0, 100.0, max_step);
    layout.lateral_bindings = vec![ChannelLateralBinding {
        boundary_id: "source".into(),
        cell_id: "cell-0".into(),
        direction,
        ownership_assertion: "Independent scalar endpoint source".into(),
    }];
    layout
}

#[test]
fn endpoint_and_lateral_exports_reserve_original_donor_water() -> TestResult {
    let shape = Shape::Rectangle {
        gravity: 1.0,
        width: 3.0,
    };
    let origin = reach(
        source_layout(shape, 1.0, ChannelLateralDirection::Withdrawal),
        initial(shape, &[1.0], &[1.0], &[0.0]),
    )?;
    let before = origin.snapshot_data()?;
    let mut forcing = interval(
        Side::Right,
        ChannelBoundary::Discharge {
            discharge_m3_per_s: 0.3,
        },
        0.0,
        0.25,
    );
    forcing.lateral_inputs = vec![ChannelLateralInput::Withdrawal {
        boundary_id: "source".into(),
        amount: WaterAmount::Rate {
            rate_m3_per_s: 12.0,
        },
    }];
    let next = origin.step(0.25, &forcing)?.next;
    assert!(next.time() > 0.0 && next.time() < 0.25);
    let receipt = next
        .last_step()?
        .ok_or("Missing competing export receipt")?;
    assert!(receipt.lateral_out_m3 + receipt.face_volume_m3[1].max(0.0) <= 3.0);
    assert_eq!(
        receipt.face_volume_m3[1].to_bits(),
        (0.3 * next.time()).to_bits()
    );
    assert!(next.time() <= receipt.depletion_limit_s);
    assert_eq!(origin.snapshot_data()?, before);
    check_water(&next)?;

    forcing.left = ChannelBoundary::Stage { elevation_m: 1.21 };
    let incoming = origin.step(0.25, &forcing)?.next;
    let receipt = incoming
        .last_step()?
        .ok_or("Missing incoming-water donor receipt")?;
    assert!(receipt.boundary_in_m3 > 0.0);
    assert!(receipt.lateral_out_m3 + receipt.boundary_out_m3 <= 3.0);
    assert_eq!(origin.snapshot_data()?, before);
    check_water(&incoming)?;
    forcing.lateral_inputs = vec![ChannelLateralInput::Withdrawal {
        boundary_id: "source".into(),
        amount: WaterAmount::IntervalVolume { volume_m3: 4.0 },
    }];
    assert!(origin.step(0.25, &forcing).is_err());
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

#[test]
fn shortened_scalar_forcing_recomputes_exact_lateral_inputs_before_acceptance() -> TestResult {
    let shape = Shape::Triangle {
        gravity: 2.0,
        coefficient: 1.0,
    };
    for volume in [0.0, 0.0001] {
        let origin = reach(
            source_layout(shape, 0.005, ChannelLateralDirection::Inflow),
            initial(shape, &[1.0], &[1.0], &[0.0]),
        )?;
        let before = origin.snapshot_data()?;
        let mut forcing = interval(
            Side::Right,
            ChannelBoundary::Stage { elevation_m: 0.81 },
            0.0,
            0.01,
        );
        let input = |amount| ChannelLateralInput::Inflow {
            boundary_id: "source".into(),
            amount,
            longitudinal_velocity_m_per_s: 0.1,
        };
        forcing.lateral_inputs = vec![input(WaterAmount::Rate {
            rate_m3_per_s: volume / 0.01,
        })];
        let rate = origin.step(0.01, &forcing)?.next;
        assert!(rate.time() > 0.0 && rate.time() <= 0.005);
        forcing.lateral_inputs = vec![input(WaterAmount::IntervalVolume { volume_m3: volume })];
        let error = origin
            .step(0.01, &forcing)
            .err()
            .ok_or("Exact forcing silently shortens")?;
        let end = match error {
            ChannelError::RecomputeInterval {
                suggested_end_s, ..
            } => suggested_end_s,
            other => return Err(format!("Unexpected exact scalar interval error: {other}").into()),
        };
        assert!(end > 0.0 && end <= 0.005);
        assert_eq!(origin.snapshot_data()?, before);
        forcing.end_s = end;
        forcing.lateral_inputs = vec![input(WaterAmount::IntervalVolume {
            volume_m3: (volume / 0.01) * end,
        })];
        let next = origin.step(end, &forcing)?.next;
        assert_eq!(next.time().to_bits(), end.to_bits());
        assert_eq!(
            next.last_step()?
                .ok_or("Missing exact scalar receipt")?
                .retries,
            0
        );
        check_water(&next)?;
    }
    Ok(())
}

#[test]
fn final_scalar_admission_shortens_source_created_supercritical_candidates() -> TestResult {
    for shape in [
        Shape::Rectangle {
            gravity: 1.0,
            width: 3.0,
        },
        Shape::Triangle {
            gravity: 2.0,
            coefficient: 1.0,
        },
    ] {
        let origin = reach(
            source_layout(shape, 0.1, ChannelLateralDirection::Inflow),
            initial(shape, &[1.0], &[1.0], &[0.0]),
        )?;
        let before = origin.snapshot_data()?;
        let mut forcing = interval(
            Side::Right,
            ChannelBoundary::Stage { elevation_m: 1.0 },
            0.0,
            0.1,
        );
        forcing.lateral_inputs = vec![ChannelLateralInput::Inflow {
            boundary_id: "source".into(),
            amount: WaterAmount::Rate {
                rate_m3_per_s: 2000.0,
            },
            longitudinal_velocity_m_per_s: 20.0,
        }];
        let next = origin.step(0.1, &forcing)?.next;
        assert!(next.time() > 0.0 && next.time() < 0.1);
        let receipt = next
            .last_step()?
            .ok_or("Missing final scalar admission receipt")?;
        let candidate = receipt
            .candidate_cfl_limit_s
            .ok_or("Missing final scalar CFL limit")?;
        assert!(next.time() <= candidate);
        let state = next.state()?;
        assert!(
            state[0]
                .velocity_m_per_s
                .ok_or("Missing wet candidate velocity")?
                .abs()
                < shape.sound(state[0].depth_m)
        );
        assert!(receipt.boundary_traces[1].is_some());
        let r = match shape {
            Shape::Rectangle { .. } => 2.0,
            Shape::Triangle { .. } => 4.0,
        };
        let final_velocity = state[0]
            .velocity_m_per_s
            .ok_or("Missing final scalar velocity")?;
        let compatible_velocity =
            final_velocity + r * (shape.sound(state[0].depth_m) - shape.sound(1.0));
        assert!(compatible_velocity.abs() < shape.sound(1.0));
        assert_eq!(origin.snapshot_data()?, before);
        check_water(&next)?;
    }
    Ok(())
}

fn continue_scalar(mut current: Reach, steps: usize) -> Result<Reach, Box<dyn std::error::Error>> {
    for _ in 0..steps {
        let start = current.time();
        let end = start + 1.0 / 1024.0;
        let forcing = ChannelForcingInterval {
            start_s: start,
            end_s: end,
            left: ChannelBoundary::Stage { elevation_m: 1.0 },
            right: ChannelBoundary::Discharge {
                discharge_m3_per_s: 0.03,
            },
            lateral_inputs: Vec::new(),
        };
        current = current.step(end, &forcing)?.next;
        assert_eq!(current.time().to_bits(), end.to_bits());
        let receipt = current
            .last_step()?
            .ok_or("Missing scalar continuation receipt")?;
        assert!(receipt.boundary_traces.iter().all(Option::is_some));
        assert_eq!(
            receipt.face_volume_m3[3].to_bits(),
            (0.03 * (end - start)).to_bits()
        );
        check_water(&current)?;
    }
    Ok(current)
}

#[test]
fn complete_scalar_restart_retains_original_targets_traces_and_config_bits() -> TestResult {
    for shape in [
        Shape::Rectangle {
            gravity: 1.0,
            width: 7.0,
        },
        Shape::Triangle {
            gravity: 2.0,
            coefficient: 1.0,
        },
    ] {
        let layout = definition(shape, 3, 4.0, 4.0, 1.0);
        let origin = reach(
            layout,
            initial(shape, &[4.0; 3], &[1.0, 1.05, 1.1], &[0.0, 0.01, 0.02]),
        )?;
        let before = origin.snapshot_data()?;
        let current = continue_scalar(origin.clone(), 3)?;
        let saved = current.snapshot_data()?;
        let trusted = origin.restore(&current.snapshot())?;
        let cold = origin.import_snapshot(&saved)?;
        assert_eq!(trusted.snapshot_data()?, saved);
        assert_eq!(cold.snapshot_data()?, saved);
        let expected = continue_scalar(current.clone(), 3)?.snapshot_data()?;
        assert_eq!(continue_scalar(trusted, 3)?.snapshot_data()?, expected);
        assert_eq!(continue_scalar(cold, 3)?.snapshot_data()?, expected);
        assert_eq!(current.snapshot_data()?, saved);
        for mutation in 0..9 {
            let mut altered = saved.clone();
            let last = altered
                .last_interval
                .as_mut()
                .ok_or("Missing saved scalar interval")?;
            match mutation {
                0 => last.forcing.left = ChannelBoundary::Stage { elevation_m: 1.01 },
                1 => {
                    last.forcing.right = ChannelBoundary::Discharge {
                        discharge_m3_per_s: f64::from_bits(0.03_f64.to_bits() + 1),
                    }
                }
                2 => {
                    last.diagnostics.boundary_traces[0]
                        .as_mut()
                        .ok_or("Missing saved trace")?
                        .depth_m += 0.01
                }
                3 => {
                    last.diagnostics.boundary_traces[1]
                        .as_mut()
                        .ok_or("Missing saved trace")?
                        .outward_discharge_m3_per_s = f64::from_bits(0.03_f64.to_bits() + 1)
                }
                4 => {
                    last.diagnostics.boundary_traces[0]
                        .as_mut()
                        .ok_or("Missing saved trace")?
                        .velocity_compatibility_abs_bound_m_per_s = f64::NAN
                }
                5 => {
                    last.diagnostics.boundary_traces[1]
                        .as_mut()
                        .ok_or("Missing saved trace")?
                        .momentum_flux_m4_per_s2 += 0.1
                }
                6 => {
                    let trace = last.diagnostics.boundary_traces[0]
                        .as_mut()
                        .ok_or("Missing saved trace")?;
                    trace.cap_compatibility_approximation = !trace.cap_compatibility_approximation;
                }
                7 => {
                    last.diagnostics.boundary_traces[1]
                        .as_mut()
                        .ok_or("Missing saved trace")?
                        .root_iterations += 1
                }
                _ => last.diagnostics.boundary_traces[1] = None,
            }
            assert!(
                origin.import_snapshot(&altered).is_err(),
                "Accepted forged scalar field {mutation}"
            );
            assert_eq!(origin.snapshot_data()?, before);
        }
        for field in 0..6 {
            let mut altered = saved.clone();
            let config = &mut altered.definition.config.boundary;
            let value = match field {
                0 => &mut config.discharge_absolute_m3_per_s,
                1 => &mut config.discharge_relative,
                2 => &mut config.momentum_absolute_m4_per_s2,
                3 => &mut config.momentum_relative,
                4 => &mut config.velocity_relative,
                _ => &mut config.critical_relative,
            };
            *value = f64::from_bits(value.to_bits() + 1);
            assert!(origin.import_snapshot(&altered).is_err());
            assert_eq!(origin.snapshot_data()?, before);
        }
    }
    Ok(())
}
