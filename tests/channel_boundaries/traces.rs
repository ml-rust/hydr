use super::common::{
    Policy, Shape, Side, TestResult, check_trace, check_water, close, definition, initial,
    interval, reach,
};
use hydr::channel::{ChannelBoundary, ChannelForcingInterval};

#[test]
fn literal_rarefaction_and_shock_traces_match_both_endpoint_orientations() -> TestResult {
    let root3 = 3.0_f64.sqrt();
    let triangle_velocity = -(95.0_f64 / 27.0).sqrt();
    let cases = [
        (
            Shape::Rectangle {
                gravity: 1.0,
                width: 3.0,
            },
            0.81,
            0.2,
            0.486,
            1.08135,
        ),
        (
            Shape::Triangle {
                gravity: 2.0,
                coefficient: 1.0,
            },
            0.81,
            0.4,
            0.26244,
            0.45927,
        ),
        (
            Shape::Rectangle {
                gravity: 1.0,
                width: 1.0,
            },
            2.0,
            -root3 / 2.0,
            -root3,
            7.0 / 2.0,
        ),
        (
            Shape::Triangle {
                gravity: 8.0,
                coefficient: 1.0,
            },
            1.5,
            triangle_velocity,
            (9.0 / 4.0) * triangle_velocity,
            203.0 / 12.0,
        ),
    ];
    for (shape, depth, velocity, outward_q, momentum_flux) in cases {
        for side in [Side::Left, Side::Right] {
            for policy in [Policy::Stage, Policy::Discharge] {
                for at_cap in [false, true] {
                    if at_cap && depth < 1.0 {
                        continue;
                    }
                    let cap = if at_cap { depth } else { 4.0 };
                    let layout = definition(shape, 1, 128.0, cap, 1.0);
                    let origin = reach(layout, initial(shape, &[128.0], &[1.0], &[0.0]))?;
                    let before = origin.snapshot_data()?;
                    let dt = 1.0 / 1024.0;
                    let forcing = interval(side, policy.boundary(side, depth, outward_q), 0.0, dt);
                    let next = origin.step(dt, &forcing)?.next;
                    assert_eq!(next.time(), dt);
                    let receipt = next
                        .last_step()?
                        .ok_or("Missing literal endpoint receipt")?;
                    let trace = receipt.boundary_traces[side.index()]
                        .as_ref()
                        .ok_or("Missing scalar trace")?;
                    assert!(receipt.boundary_traces[1 - side.index()].is_none());
                    close(trace.depth_m, depth);
                    close(trace.area_m2, shape.area(depth));
                    close(trace.outward_velocity_m_per_s, velocity);
                    close(trace.outward_discharge_m3_per_s, outward_q);
                    close(trace.momentum_flux_m4_per_s2, momentum_flux);
                    assert!(trace.depth_m <= cap);
                    if trace.cap_compatibility_approximation {
                        assert!(matches!(policy, Policy::Discharge));
                        assert!(at_cap);
                        assert_eq!(trace.depth_m.to_bits(), cap.to_bits());
                    }
                    check_trace(trace, next.config(), shape, 1.0);
                    if trace.cap_compatibility_approximation {
                        let ai = shape.area(1.0);
                        let ab = shape.area(trace.depth_m);
                        let curve_velocity =
                            -((shape.pressure(trace.depth_m) - shape.pressure(1.0)) * (ab - ai)
                                / (ab * ai))
                                .sqrt();
                        assert!(curve_velocity.abs() < shape.sound(trace.depth_m));
                        let residual = (trace.outward_velocity_m_per_s - curve_velocity).abs();
                        assert!(residual <= trace.velocity_compatibility_abs_bound_m_per_s);
                        assert!(ab * residual <= trace.water_compatibility_abs_bound_m3_per_s);
                        assert!(
                            ab * residual * (2.0 * trace.outward_velocity_m_per_s.abs() + residual)
                                <= trace.momentum_compatibility_abs_bound_m4_per_s2
                        );
                    }
                    let state = next.state()?;
                    close(state[0].volume_m3, shape.area(1.0) * 128.0 - outward_q * dt);
                    close(
                        state[0].longitudinal_momentum_m4_per_s,
                        side.sign() * (shape.pressure(1.0) - momentum_flux) * dt,
                    );
                    if matches!(policy, Policy::Discharge) {
                        assert_eq!(
                            receipt.face_volume_m3[side.face(1)].to_bits(),
                            (side.sign() * outward_q * dt).to_bits()
                        );
                        assert_eq!(trace.water_flux_width_m3_per_s.to_bits(), 0.0_f64.to_bits());
                    }
                    assert_eq!(origin.snapshot_data()?, before);
                    check_water(&next)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn two_scalar_endpoints_use_physical_pressure_in_the_one_cell_recurrence() -> TestResult {
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
            definition(shape, 1, 16.0, 4.0, 1.0),
            initial(shape, &[16.0], &[1.0], &[0.0]),
        )?;
        let dt = 1.0 / 1024.0;
        let forcing = ChannelForcingInterval {
            start_s: 0.0,
            end_s: dt,
            left: ChannelBoundary::Stage { elevation_m: 1.1 },
            right: ChannelBoundary::Stage { elevation_m: 0.9 },
            lateral_inputs: Vec::new(),
        };
        let next = origin.step(dt, &forcing)?.next;
        let receipt = next.last_step()?.ok_or("Missing pressure recurrence")?;
        let left = receipt.boundary_traces[0]
            .as_ref()
            .ok_or("Missing left scalar trace")?;
        let right = receipt.boundary_traces[1]
            .as_ref()
            .ok_or("Missing right scalar trace")?;
        let state = next.state()?;
        close(
            state[0].longitudinal_momentum_m4_per_s,
            dt * (left.momentum_flux_m4_per_s2 - right.momentum_flux_m4_per_s2),
        );
        close(
            state[0].volume_m3,
            shape.area(1.0) * 16.0
                - dt * (left.outward_discharge_m3_per_s + right.outward_discharge_m3_per_s),
        );
        match shape {
            Shape::Triangle { .. } => {
                let certificate = receipt
                    .riemann
                    .as_ref()
                    .ok_or("Missing scalar triangle certificates")?;
                assert_eq!(
                    certificate.water_flux_width_m3_per_s,
                    vec![
                        left.water_flux_width_m3_per_s,
                        right.water_flux_width_m3_per_s
                    ]
                );
                assert_eq!(
                    certificate.momentum_flux_width_m4_per_s2,
                    vec![
                        left.momentum_flux_width_m4_per_s2,
                        right.momentum_flux_width_m4_per_s2
                    ]
                );
                assert_eq!(
                    certificate.root_iterations,
                    left.root_iterations.max(right.root_iterations)
                );
            }
            Shape::Rectangle { .. } => assert!(receipt.riemann.is_none()),
        }
        check_water(&next)?;
    }
    Ok(())
}

#[test]
fn original_longitudinal_discharge_bits_survive_width_round_trips_and_signed_zero() -> TestResult {
    for (width, target) in [(7.0, 0.03_f64), (11.0, 0.1)] {
        assert_ne!(((target / width) * width).to_bits(), target.to_bits());
        for q in [target, -target, 0.0, -0.0] {
            let shape = Shape::Rectangle {
                gravity: 1.0,
                width,
            };
            let origin = reach(
                definition(shape, 1, 16.0, 4.0, 1.0),
                initial(shape, &[16.0], &[1.0], &[q / width]),
            )?;
            let dt = 1.0 / 1024.0;
            let boundary = ChannelBoundary::Discharge {
                discharge_m3_per_s: q,
            };
            let forcing = ChannelForcingInterval {
                start_s: 0.0,
                end_s: dt,
                left: boundary.clone(),
                right: boundary,
                lateral_inputs: Vec::new(),
            };
            let next = origin.step(dt, &forcing)?.next;
            let receipt = next
                .last_step()?
                .ok_or("Missing original discharge receipt")?;
            for face in [0, 1] {
                assert_eq!(receipt.face_volume_m3[face].to_bits(), (q * dt).to_bits());
                assert_eq!(
                    receipt.boundary_traces[face]
                        .as_ref()
                        .ok_or("Missing discharge trace")?
                        .water_flux_width_m3_per_s
                        .to_bits(),
                    0.0_f64.to_bits()
                );
            }
            let saved = next.snapshot_data()?;
            let interval = &saved
                .last_interval
                .as_ref()
                .ok_or("Missing saved discharge interval")?
                .forcing;
            for boundary in [&interval.left, &interval.right] {
                let ChannelBoundary::Discharge { discharge_m3_per_s } = boundary else {
                    return Err("Saved discharge changes its policy".into());
                };
                assert_eq!(discharge_m3_per_s.to_bits(), q.to_bits());
            }
            assert_eq!(origin.import_snapshot(&saved)?.snapshot_data()?, saved);
            check_water(&next)?;
        }
    }
    Ok(())
}

#[test]
fn stage_elevation_translation_preserves_hydraulics_across_datum_shifts() -> TestResult {
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
        for side in [Side::Left, Side::Right] {
            let mut expected = None;
            for bed in [0.0, 128.0, -128.0] {
                let mut layout = definition(shape, 1, 128.0, 4.0, 1.0);
                match &mut layout.geometry {
                    hydr::channel::ChannelGeometry::Rectangular { cells } => {
                        cells[0].bed_elevation_m = bed;
                    }
                    hydr::channel::ChannelGeometry::PrismaticTriangular {
                        bed_elevation_m, ..
                    } => {
                        *bed_elevation_m = bed;
                    }
                }
                let origin = reach(layout, initial(shape, &[128.0], &[1.0], &[0.0]))?;
                let before = origin.snapshot_data()?;
                let dt = 1.0 / 1024.0;
                let elevation_m = bed + 0.75;
                assert_eq!((elevation_m - bed).to_bits(), 0.75_f64.to_bits());
                let forcing = interval(side, ChannelBoundary::Stage { elevation_m }, 0.0, dt);
                let next = origin.step(dt, &forcing)?.next;
                assert_eq!(next.time().to_bits(), dt.to_bits());
                let receipt = next
                    .last_step()?
                    .ok_or("Missing translated-stage receipt")?;
                let trace = receipt.boundary_traces[side.index()]
                    .as_ref()
                    .ok_or("Missing translated-stage trace")?;
                assert_eq!(trace.depth_m.to_bits(), 0.75_f64.to_bits());
                check_trace(trace, next.config(), shape, 1.0);
                let state = next.state()?;
                let cell = &state[0];
                let hydraulic = (
                    [
                        trace.depth_m.to_bits(),
                        trace.area_m2.to_bits(),
                        trace.outward_velocity_m_per_s.to_bits(),
                        trace.outward_discharge_m3_per_s.to_bits(),
                        trace.momentum_flux_m4_per_s2.to_bits(),
                    ],
                    [
                        cell.volume_m3.to_bits(),
                        cell.depth_m.to_bits(),
                        cell.longitudinal_momentum_m4_per_s.to_bits(),
                        cell.discharge_m3_per_s.to_bits(),
                        cell.velocity_m_per_s
                            .ok_or("Missing translated-stage wet velocity")?
                            .to_bits(),
                    ],
                    cell.dry,
                    next.ledger().clone(),
                );
                if let Some(expected) = &expected {
                    assert_eq!(&hydraulic, expected);
                } else {
                    expected = Some(hydraulic);
                }
                assert_eq!(origin.snapshot_data()?, before);
                check_water(&next)?;
            }
        }
    }
    Ok(())
}
