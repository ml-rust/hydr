use super::common::{Shape, Side, TestResult, definition, initial, interval, reach};
use hydr::channel::{ChannelBoundary, ChannelConfig};

#[test]
fn scalar_endpoints_reject_dry_critical_supercritical_and_out_of_range_states_atomically()
-> TestResult {
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
        let (critical_depth, critical_q) = match shape {
            Shape::Rectangle { .. } => (4.0 / 9.0, 8.0 / 9.0),
            Shape::Triangle { .. } => (16.0 / 25.0, 1024.0 / 3125.0),
        };
        for side in [Side::Left, Side::Right] {
            for (h, u, boundary) in [
                (0.0, 0.0, ChannelBoundary::Stage { elevation_m: 1.0 }),
                (
                    0.0,
                    0.0,
                    ChannelBoundary::Discharge {
                        discharge_m3_per_s: 0.0,
                    },
                ),
                (
                    1.0,
                    side.sign(),
                    ChannelBoundary::Stage { elevation_m: 1.0 },
                ),
                (
                    1.0,
                    2.0 * side.sign(),
                    ChannelBoundary::Discharge {
                        discharge_m3_per_s: 0.0,
                    },
                ),
                (
                    1.0,
                    0.0,
                    ChannelBoundary::Stage {
                        elevation_m: critical_depth,
                    },
                ),
                (
                    1.0,
                    0.0,
                    ChannelBoundary::Discharge {
                        discharge_m3_per_s: side.sign() * critical_q,
                    },
                ),
                (1.0, 0.0, ChannelBoundary::Stage { elevation_m: 0.0 }),
                (1.0, 0.0, ChannelBoundary::Stage { elevation_m: 4.1 }),
                (
                    1.0,
                    0.0,
                    ChannelBoundary::Stage {
                        elevation_m: f64::NAN,
                    },
                ),
                (
                    1.0,
                    0.0,
                    ChannelBoundary::Discharge {
                        discharge_m3_per_s: side.sign() * 100.0,
                    },
                ),
                (
                    1.0,
                    0.0,
                    ChannelBoundary::Discharge {
                        discharge_m3_per_s: f64::INFINITY,
                    },
                ),
            ] {
                let origin = reach(
                    definition(shape, 1, 16.0, 4.0, 1.0),
                    initial(shape, &[16.0], &[h], &[u]),
                )?;
                let before = origin.snapshot_data()?;
                assert!(
                    origin
                        .step(
                            1.0 / 1024.0,
                            &interval(side, boundary.clone(), 0.0, 1.0 / 1024.0)
                        )
                        .is_err(),
                    "Accepted unsupported scalar boundary: {shape:?}, {side:?}, h={h}, u={u}, {boundary:?}"
                );
                assert_eq!(origin.snapshot_data()?, before);
            }
        }
    }
    Ok(())
}

#[test]
fn scalar_discharge_rejects_certified_targets_beyond_the_original_cap() -> TestResult {
    for shape in [
        Shape::Rectangle {
            gravity: 1.0,
            width: 1.0,
        },
        Shape::Triangle {
            gravity: 8.0,
            coefficient: 1.0,
        },
    ] {
        let cap = match shape {
            Shape::Rectangle { .. } => 2.0,
            _ => 1.5,
        };
        let h = cap + 0.01;
        let ai = shape.area(1.0);
        let ab = shape.area(h);
        let velocity = -((shape.pressure(h) - shape.pressure(1.0)) * (ab - ai) / (ab * ai)).sqrt();
        for side in [Side::Left, Side::Right] {
            let origin = reach(
                definition(shape, 1, 128.0, cap, 1.0),
                initial(shape, &[128.0], &[1.0], &[0.0]),
            )?;
            let before = origin.snapshot_data()?;
            let forcing = interval(
                side,
                ChannelBoundary::Discharge {
                    discharge_m3_per_s: side.sign() * ab * velocity,
                },
                0.0,
                1.0 / 1024.0,
            );
            assert!(origin.step(1.0 / 1024.0, &forcing).is_err());
            assert_eq!(origin.snapshot_data()?, before);
        }
    }
    Ok(())
}

#[test]
fn boundary_accuracy_config_rejects_nonfinite_and_degenerate_controls() -> TestResult {
    let shape = Shape::Rectangle {
        gravity: 1.0,
        width: 3.0,
    };
    let defaults = ChannelConfig::default().boundary;
    assert_eq!(defaults.discharge_absolute_m3_per_s, 1e-12);
    assert_eq!(defaults.discharge_relative, 1e-12);
    assert_eq!(defaults.momentum_absolute_m4_per_s2, 1e-12);
    assert_eq!(defaults.momentum_relative, 1e-12);
    assert_eq!(defaults.velocity_relative, 1e-12);
    assert_eq!(defaults.critical_relative, 64.0 * f64::EPSILON);
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        for field in 0..6 {
            let mut layout = definition(shape, 1, 16.0, 4.0, 1.0);
            match field {
                0 => layout.config.boundary.discharge_absolute_m3_per_s = invalid,
                1 => layout.config.boundary.discharge_relative = invalid,
                2 => layout.config.boundary.momentum_absolute_m4_per_s2 = invalid,
                3 => layout.config.boundary.momentum_relative = invalid,
                4 => layout.config.boundary.velocity_relative = invalid,
                _ => layout.config.boundary.critical_relative = invalid,
            }
            assert!(reach(layout, initial(shape, &[16.0], &[1.0], &[0.0])).is_err());
        }
    }
    for field in 0..5 {
        let mut layout = definition(shape, 1, 16.0, 4.0, 1.0);
        match field {
            0 => {
                layout.config.boundary.discharge_absolute_m3_per_s = 0.0;
                layout.config.boundary.discharge_relative = 0.0;
            }
            1 => {
                layout.config.boundary.momentum_absolute_m4_per_s2 = 0.0;
                layout.config.boundary.momentum_relative = 0.0;
            }
            2 => layout.config.boundary.velocity_relative = 0.0,
            3 => layout.config.boundary.critical_relative = 0.0,
            _ => layout.config.boundary.critical_relative = 1.0,
        }
        assert!(reach(layout, initial(shape, &[16.0], &[1.0], &[0.0])).is_err());
    }
    Ok(())
}

#[test]
fn a_nonzero_prescribed_transfer_cannot_disappear_under_duration_multiplication() -> TestResult {
    let shape = Shape::Rectangle {
        gravity: 1.0,
        width: 1.0,
    };
    for side in [Side::Left, Side::Right] {
        let origin = reach(
            definition(shape, 1, 128.0, 4.0, 1.0),
            initial(shape, &[128.0], &[1.0], &[0.0]),
        )?;
        let before = origin.snapshot_data()?;
        let q = side.sign() * f64::from_bits(1);
        assert_eq!(q * 0.5, 0.0);
        let forcing = interval(
            side,
            ChannelBoundary::Discharge {
                discharge_m3_per_s: q,
            },
            0.0,
            0.5,
        );
        assert!(origin.step(0.5, &forcing).is_err());
        assert_eq!(origin.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn represented_depth_equality_cannot_claim_unresolved_scalar_flux_precision() -> TestResult {
    let shape = Shape::Triangle {
        gravity: 8.0,
        coefficient: 1.0,
    };
    let depth = 2.0_f64.sqrt();
    let dt = 1.0 / 1024.0;
    let initial = hydr::channel::ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![32.0],
        longitudinal_momentum_m4_per_s: vec![0.0],
    };
    let stage = ChannelBoundary::Stage { elevation_m: depth };
    let forcing = hydr::channel::ChannelForcingInterval {
        start_s: 0.0,
        end_s: dt,
        left: stage.clone(),
        right: stage,
        lateral_inputs: Vec::new(),
    };
    let default = reach(definition(shape, 1, 16.0, 4.0, 1.0), initial.clone())?;
    assert_eq!(default.state()?[0].depth_m.to_bits(), depth.to_bits());
    let accepted = default.step(dt, &forcing)?.next;
    let receipt = accepted
        .last_step()?
        .ok_or("Missing represented-depth scalar receipt")?;
    for trace in receipt.boundary_traces.iter() {
        let trace = trace
            .as_ref()
            .ok_or("Missing represented-depth scalar trace")?;
        assert!(trace.water_flux_width_m3_per_s > 1e-18);
        super::common::check_trace(trace, accepted.config(), shape, depth);
    }
    super::common::check_water(&accepted)?;

    let mut layout = definition(shape, 1, 16.0, 4.0, 1.0);
    layout.config.riemann_water_flux_absolute_m3_per_s = 1e-18;
    layout.config.riemann_flux_relative = 0.0;
    let strict = reach(layout, initial)?;
    let before = strict.snapshot_data()?;
    let error = strict
        .step(dt, &forcing)
        .err()
        .ok_or("Scalar equality claims unsupported flux precision")?;
    assert!(matches!(
        error,
        hydr::channel::ChannelError::Numerical { .. }
    ));
    assert_eq!(strict.snapshot_data()?, before);
    Ok(())
}
