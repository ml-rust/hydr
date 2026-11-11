use super::boundary_common::{
    Side, TestResult, check_trace, check_water, definition, initial, interval, reach,
};
use super::fixtures::{FIXTURES, reservoir};
use hydr::channel::{ChannelBoundary, ChannelForcingInterval, ChannelGeometry};

fn accuracy(actual: f64, expected: f64, absolute: f64) {
    assert!(
        (actual - expected).abs() <= absolute + 1e-12 * expected.abs(),
        "actual={actual:e}, independent reference={expected:e}"
    );
}

#[test]
fn frozen_exact_input_roots_fluxes_and_passivity_match_both_ends() -> TestResult {
    for fixture in FIXTURES {
        for side in [Side::Left, Side::Right] {
            let origin = reach(
                definition(fixture.shape, 1, 128.0, 4.0, 1.0),
                initial(fixture.shape, &[128.0], &[1.0], &[0.0]),
            )?;
            let before = origin.snapshot_data()?;
            let dt = 1.0 / 1024.0;
            let next = origin
                .step(dt, &interval(side, fixture.boundary(), 0.0, dt))?
                .next;
            let receipt = next.last_step()?.ok_or("Missing reservoir receipt")?;
            let trace = receipt.boundary_traces[side.index()]
                .as_ref()
                .ok_or("Missing reservoir trace")?;
            let certificate = trace
                .reservoir
                .as_ref()
                .ok_or("Missing reservoir certificate")?;
            accuracy(trace.depth_m, fixture.depth, 1e-12);
            accuracy(trace.outward_velocity_m_per_s, fixture.velocity, 1e-12);
            accuracy(trace.outward_discharge_m3_per_s, fixture.discharge, 1e-12);
            accuracy(trace.momentum_flux_m4_per_s2, fixture.momentum, 1e-12);
            assert!(
                certificate.depth_lower_m <= f64::from_bits(fixture.root_floor_bits),
                "{}",
                fixture.name
            );
            assert!(
                certificate.depth_upper_m >= f64::from_bits(fixture.root_ceil_bits),
                "{}",
                fixture.name
            );
            assert!(certificate.depth_lower_m <= trace.depth_m);
            assert!(trace.depth_m <= certificate.depth_upper_m);
            assert!(!trace.cap_compatibility_approximation);
            let u = trace.outward_velocity_m_per_s;
            let q = trace.outward_discharge_m3_per_s;
            let gravity = fixture.shape.gravity();
            let velocity_head = u * u / (2.0 * gravity);
            let incoming_head = u.min(0.0).powi(2) / (2.0 * gravity);
            let head_residual =
                trace.depth_m + (1.0 + fixture.loss) * incoming_head - fixture.elevation;
            let arithmetic = 64.0
                * f64::EPSILON
                * (trace.depth_m.abs() + fixture.elevation.abs() + velocity_head);
            assert!(head_residual.abs() <= certificate.head_compatibility_abs_bound_m + arithmetic);
            let scale = trace
                .depth_m
                .abs()
                .max(fixture.elevation.abs())
                .max((1.0 + fixture.loss) * incoming_head);
            assert!(
                certificate.head_compatibility_abs_bound_m
                    <= next.config().boundary.reservoir_head_absolute_m
                        + next.config().boundary.reservoir_head_relative * scale
            );
            let dissipation = q * gravity * (trace.depth_m - fixture.elevation + velocity_head);
            accuracy(dissipation, fixture.dissipation, 1e-10);
            let model_dissipation = 0.5 * u * u * (q.max(0.0) + fixture.loss * (-q).max(0.0));
            let error_bound =
                gravity * q.abs() * (certificate.head_compatibility_abs_bound_m + arithmetic);
            assert!(
                (dissipation - model_dissipation).abs()
                    <= error_bound + 64.0 * f64::EPSILON * model_dissipation.abs()
            );
            assert!(dissipation >= -error_bound);
            assert!(certificate.dissipation_head_lower_m.is_finite());
            assert!(certificate.dissipation_head_upper_m.is_finite());
            assert!(certificate.dissipation_head_lower_m <= certificate.dissipation_head_upper_m);
            let passive_head = if u >= 0.0 {
                velocity_head
            } else {
                fixture.loss * velocity_head
            };
            assert!(certificate.dissipation_head_lower_m <= passive_head + arithmetic);
            assert!(certificate.dissipation_head_upper_m + arithmetic >= passive_head);
            accuracy(
                receipt.face_volume_m3[side.face(1)],
                side.sign() * q * dt,
                1e-12,
            );
            check_trace(trace, next.config(), fixture.shape, 1.0);
            check_water(&next)?;
            assert_eq!(origin.snapshot_data()?, before);
        }
    }
    Ok(())
}

#[test]
fn outgoing_trace_is_independent_of_every_finite_entrance_loss() -> TestResult {
    for fixture in FIXTURES.into_iter().take(2) {
        for side in [Side::Left, Side::Right] {
            let origin = reach(
                definition(fixture.shape, 1, 128.0, 4.0, 1.0),
                initial(fixture.shape, &[128.0], &[1.0], &[0.0]),
            )?;
            let dt = 1.0 / 1024.0;
            let mut baseline = None;
            for loss in [0.0, -0.0, 1.0, 1e100, f64::MAX] {
                let next = origin
                    .step(
                        dt,
                        &interval(side, reservoir(fixture.elevation, loss), 0.0, dt),
                    )?
                    .next;
                let receipt = next
                    .last_step()?
                    .ok_or("Missing outgoing reservoir receipt")?;
                let trace = receipt.boundary_traces[side.index()]
                    .as_ref()
                    .ok_or("Missing outgoing reservoir trace")?;
                if let Some(expected) = &baseline {
                    assert_eq!(trace, expected);
                } else {
                    baseline = Some(trace.clone());
                }
            }
        }
    }
    Ok(())
}

#[test]
fn resolved_reversal_converges_continuously_toward_zero_discharge() -> TestResult {
    for fixture in FIXTURES.into_iter().take(2) {
        for side in [Side::Left, Side::Right] {
            let origin = reach(
                definition(fixture.shape, 1, 128.0, 4.0, 1.0),
                initial(fixture.shape, &[128.0], &[1.0], &[0.0]),
            )?;
            for loss in [0.0, 1.0] {
                let mut previous = [f64::INFINITY; 2];
                for exponent in [8, 12, 16, 20, 24] {
                    let offset = 2.0_f64.powi(-exponent);
                    for (index, sign) in [-1.0, 1.0].into_iter().enumerate() {
                        let dt = 1.0 / 1024.0;
                        let next = origin
                            .step(
                                dt,
                                &interval(side, reservoir(1.0 + sign * offset, loss), 0.0, dt),
                            )?
                            .next;
                        let receipt = next.last_step()?.ok_or("Missing reversal receipt")?;
                        let trace = receipt.boundary_traces[side.index()]
                            .as_ref()
                            .ok_or("Missing reversal trace")?;
                        let q = trace.outward_discharge_m3_per_s;
                        assert!(q * sign < 0.0);
                        assert!(q.abs() < previous[index]);
                        assert!(q.abs() <= 8.0 * fixture.shape.area(1.0) * offset);
                        previous[index] = q.abs();
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn exact_datum_shift_preserves_local_physics_and_head_budget() -> TestResult {
    let fixture = FIXTURES[2];
    for side in [Side::Left, Side::Right] {
        let mut physical = None;
        for bed in [0.0, 16.0] {
            let mut layout = definition(fixture.shape, 1, 128.0, 4.0, 1.0);
            match &mut layout.geometry {
                ChannelGeometry::Rectangular { cells } => cells[0].bed_elevation_m = bed,
                _ => return Err("Datum fixture changes geometry".into()),
            }
            let origin = reach(layout, initial(fixture.shape, &[128.0], &[1.0], &[0.0]))?;
            let dt = 1.0 / 1024.0;
            let next = origin
                .step(dt, &interval(side, reservoir(bed + 2.375, 0.0), 0.0, dt))?
                .next;
            let receipt = next.last_step()?.ok_or("Missing datum receipt")?;
            let trace = receipt.boundary_traces[side.index()]
                .as_ref()
                .ok_or("Missing datum trace")?;
            if let Some(expected) = &physical {
                assert_eq!(trace, expected);
            } else {
                physical = Some(trace.clone());
            }
        }
    }
    Ok(())
}

#[test]
fn every_stage_discharge_reservoir_pair_retains_endpoint_identity() -> TestResult {
    for fixture in FIXTURES.into_iter().take(2) {
        let origin = reach(
            definition(fixture.shape, 3, 128.0, 4.0, 1.0),
            initial(fixture.shape, &[128.0; 3], &[1.0; 3], &[0.0; 3]),
        )?;
        let policies = [
            ChannelBoundary::Stage { elevation_m: 0.81 },
            ChannelBoundary::Discharge {
                discharge_m3_per_s: 0.0,
            },
            fixture.boundary(),
        ];
        for left in &policies {
            for right in &policies {
                let dt = 1.0 / 1024.0;
                let forcing = ChannelForcingInterval {
                    start_s: 0.0,
                    end_s: dt,
                    left: left.clone(),
                    right: right.clone(),
                    lateral_inputs: Vec::new(),
                };
                let next = origin.step(dt, &forcing)?.next;
                let receipt = next.last_step()?.ok_or("Missing mixed policy receipt")?;
                for (side, policy) in [left, right].into_iter().enumerate() {
                    let trace = receipt.boundary_traces[side]
                        .as_ref()
                        .ok_or("Missing mixed policy trace")?;
                    assert_eq!(
                        trace.reservoir.is_some(),
                        matches!(policy, ChannelBoundary::QuiescentReservoir { .. })
                    );
                    if matches!(policy, ChannelBoundary::QuiescentReservoir { .. }) {
                        accuracy(trace.depth_m, 0.81, 1e-12);
                    }
                }
                check_water(&next)?;
            }
        }
    }
    Ok(())
}

#[test]
fn reservoir_candidates_preserve_postfriction_admission() -> TestResult {
    for fixture in FIXTURES.into_iter().take(2) {
        for side in [Side::Left, Side::Right] {
            let mut layout = definition(fixture.shape, 1, 128.0, 4.0, 1.0);
            match &mut layout.geometry {
                ChannelGeometry::Rectangular { cells } => cells[0].manning_n = 0.03,
                ChannelGeometry::PrismaticTriangular { cells, .. } => cells[0].manning_n = 0.03,
            }
            let origin = reach(
                layout,
                initial(fixture.shape, &[128.0], &[1.0], &[0.1 * side.sign()]),
            )?;
            let before = origin.snapshot_data()?;
            let dt = 1.0 / 1024.0;
            let next = origin
                .step(dt, &interval(side, fixture.boundary(), 0.0, dt))?
                .next;
            let receipt = next
                .last_step()?
                .ok_or("Missing reservoir friction receipt")?;
            assert!(receipt.friction_impulse_m4_per_s[0].abs() > 0.0);
            let state = next.state()?;
            assert!(
                state[0]
                    .velocity_m_per_s
                    .ok_or("Missing postfriction velocity")?
                    .abs()
                    < fixture.shape.sound(state[0].depth_m)
            );
            let continued = next
                .step(2.0 * dt, &interval(side, fixture.boundary(), dt, 2.0 * dt))?
                .next;
            check_water(&continued)?;
            assert_eq!(origin.snapshot_data()?, before);
        }
    }
    Ok(())
}

#[test]
fn reservoir_head_equal_to_cap_admits_the_strictly_interior_incoming_root() -> TestResult {
    let shape = super::boundary_common::Shape::Rectangle {
        gravity: 1.0,
        width: 1.0,
    };
    for side in [Side::Left, Side::Right] {
        let origin = reach(
            definition(shape, 1, 128.0, 2.0, 1.0),
            initial(shape, &[128.0], &[1.0], &[0.0]),
        )?;
        let before = origin.snapshot_data()?;
        let dt = 1.0 / 1024.0;
        let next = origin
            .step(dt, &interval(side, reservoir(2.0, 0.0), 0.0, dt))?
            .next;
        let receipt = next
            .last_step()?
            .ok_or("Missing cap-head reservoir receipt")?;
        let trace = receipt.boundary_traces[side.index()]
            .as_ref()
            .ok_or("Missing cap-head reservoir trace")?;
        let certificate = trace
            .reservoir
            .as_ref()
            .ok_or("Missing cap-head reservoir certificate")?;
        assert!(1.0 < certificate.depth_lower_m);
        assert!(certificate.depth_lower_m <= trace.depth_m);
        assert!(trace.depth_m <= certificate.depth_upper_m);
        assert!(certificate.depth_upper_m < 2.0);
        assert!(trace.outward_velocity_m_per_s < 0.0);
        assert!(!trace.cap_compatibility_approximation);
        for (depth, lower) in [
            (certificate.depth_lower_m, true),
            (certificate.depth_upper_m, false),
        ] {
            let velocity_squared = 0.5 * (depth * depth - 1.0) * (depth - 1.0) / depth;
            let residual = depth + 0.5 * velocity_squared - 2.0;
            let arithmetic = 32.0 * f64::EPSILON;
            if lower {
                assert!(residual <= arithmetic);
            } else {
                assert!(residual >= -arithmetic);
            }
        }
        let head_residual = trace.depth_m + trace.outward_velocity_m_per_s.powi(2) / 2.0 - 2.0;
        assert!(
            head_residual.abs() <= certificate.head_compatibility_abs_bound_m + 32.0 * f64::EPSILON
        );
        check_trace(trace, next.config(), shape, 1.0);
        check_water(&next)?;
        assert_eq!(origin.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn equal_reservoir_head_preserves_lake_rest_with_complete_receipts() -> TestResult {
    for fixture in FIXTURES.into_iter().take(2) {
        for side in [Side::Left, Side::Right] {
            for loss in [0.0, 1.0] {
                let origin = reach(
                    definition(fixture.shape, 3, 4.0, 4.0, 0.25),
                    initial(fixture.shape, &[4.0; 3], &[1.0; 3], &[0.0; 3]),
                )?;
                let before = origin.snapshot_data()?;
                let initial_storage = origin.ledger().storage_m3;
                let mut current = origin.clone();
                while current.time() < 2.0 {
                    let start = current.time();
                    current = current
                        .step(2.0, &interval(side, reservoir(1.0, loss), start, 2.0))?
                        .next;
                    assert!(current.time() > start && current.time() <= 2.0);
                    let receipt = current
                        .last_step()?
                        .ok_or("Missing reservoir rest receipt")?;
                    let trace = receipt.boundary_traces[side.index()]
                        .as_ref()
                        .ok_or("Missing reservoir rest trace")?;
                    let certificate = trace
                        .reservoir
                        .as_ref()
                        .ok_or("Missing reservoir rest certificate")?;
                    assert!(receipt.boundary_traces[1 - side.index()].is_none());
                    assert!(certificate.depth_lower_m <= 1.0);
                    assert!(certificate.depth_upper_m >= 1.0);
                    assert!((trace.depth_m - 1.0).abs() <= 1e-12);
                    assert!(trace.outward_velocity_m_per_s.abs() <= 1e-11);
                    assert!(trace.outward_discharge_m3_per_s.abs() <= 1e-11);
                    assert!(
                        (trace.momentum_flux_m4_per_s2 - fixture.shape.pressure(1.0)).abs()
                            <= 1e-11
                    );
                    assert!(!trace.cap_compatibility_approximation);
                    let incoming_speed = (-trace.outward_velocity_m_per_s).max(0.0);
                    let head_residual = trace.depth_m
                        + (1.0 + loss) * incoming_speed * incoming_speed
                            / (2.0 * fixture.shape.gravity())
                        - 1.0;
                    assert!(
                        head_residual.abs()
                            <= certificate.head_compatibility_abs_bound_m + 32.0 * f64::EPSILON
                    );
                    assert!(certificate.head_compatibility_abs_bound_m >= 0.0);
                    assert!(
                        certificate.head_compatibility_abs_bound_m
                            <= current.config().boundary.reservoir_head_absolute_m
                                + current.config().boundary.reservoir_head_relative
                                    * trace.depth_m.max(1.0)
                    );
                    assert!(certificate.dissipation_head_lower_m.is_finite());
                    assert!(certificate.dissipation_head_upper_m.is_finite());
                    assert!(certificate.dissipation_head_lower_m >= 0.0);
                    assert!(
                        certificate.dissipation_head_lower_m
                            <= certificate.dissipation_head_upper_m
                    );
                    assert!(certificate.dissipation_head_upper_m <= 1e-11);
                    assert!(
                        receipt
                            .face_volume_m3
                            .iter()
                            .all(|volume| volume.abs() <= 1e-11)
                    );
                    for cell in current.state()? {
                        assert!((cell.depth_m - 1.0).abs() <= 1e-12);
                        assert!(cell.discharge_m3_per_s.abs() <= 1e-11);
                        assert!(cell.longitudinal_momentum_m4_per_s.abs() <= 1e-11);
                    }
                    assert!((current.ledger().storage_m3 - initial_storage).abs() <= 1e-11);
                    assert!(current.ledger().boundary_in_m3.abs() <= 1e-11);
                    assert!(current.ledger().boundary_out_m3.abs() <= 1e-11);
                    check_trace(trace, current.config(), fixture.shape, 1.0);
                    check_water(&current)?;
                }
                assert_eq!(origin.snapshot_data()?, before);
            }
        }
    }
    Ok(())
}
