use super::common::{
    Reach, Shape, Side, TestResult, check_trace, check_water, close, definition, initial, reach,
};
use hydr::channel::{ChannelBoundary, ChannelForcingInterval};

fn frame(side: Side, special: ChannelBoundary, ordinary_outward_q: f64) -> ChannelForcingInterval {
    let mut ends = [
        ChannelBoundary::Discharge {
            discharge_m3_per_s: -ordinary_outward_q,
        },
        ChannelBoundary::Discharge {
            discharge_m3_per_s: ordinary_outward_q,
        },
    ];
    ends[side.index()] = special;
    ChannelForcingInterval {
        start_s: 0.0,
        end_s: 1.0 / 1024.0,
        left: ends[0].clone(),
        right: ends[1].clone(),
        lateral_inputs: Vec::new(),
    }
}

fn check_mapping(current: &Reach, forcing: &ChannelForcingInterval, shape: Shape) -> TestResult {
    let receipt = current
        .last_step()?
        .ok_or("Missing packed scalar receipt")?;
    let dt = receipt.end_s - receipt.start_s;
    let saved = current.snapshot_data()?;
    let retained = &saved
        .last_interval
        .as_ref()
        .ok_or("Missing packed saved interval")?
        .forcing;
    for (endpoint, (boundary, original)) in [&forcing.left, &forcing.right]
        .into_iter()
        .zip([&retained.left, &retained.right])
        .enumerate()
    {
        let trace = receipt.boundary_traces[endpoint]
            .as_ref()
            .ok_or("Missing mapped scalar trace")?;
        check_trace(trace, current.config(), shape, 1.0);
        match boundary {
            ChannelBoundary::Stage { elevation_m } => {
                close(trace.depth_m, *elevation_m);
                assert_eq!(trace.root_iterations, 0);
                assert!(!trace.cap_compatibility_approximation);
                let ChannelBoundary::Stage {
                    elevation_m: retained,
                } = original
                else {
                    return Err("Packed Stage changes endpoint policy".into());
                };
                assert_eq!(retained.to_bits(), elevation_m.to_bits());
            }
            ChannelBoundary::Discharge { discharge_m3_per_s } => {
                let sign = if endpoint == 0 { -1.0 } else { 1.0 };
                assert_eq!(
                    trace.outward_discharge_m3_per_s.to_bits(),
                    (sign * discharge_m3_per_s).to_bits()
                );
                assert_eq!(
                    receipt.face_volume_m3[endpoint].to_bits(),
                    (discharge_m3_per_s * dt).to_bits()
                );
                assert_eq!(trace.water_flux_width_m3_per_s.to_bits(), 0.0_f64.to_bits());
                let ChannelBoundary::Discharge {
                    discharge_m3_per_s: retained,
                } = original
                else {
                    return Err("Packed Discharge changes endpoint policy".into());
                };
                assert_eq!(retained.to_bits(), discharge_m3_per_s.to_bits());
            }
            _ => return Err("Packed fixture requires two scalar endpoints".into()),
        }
    }
    check_water(current)
}

fn check_restart(
    origin: &Reach,
    current: Reach,
    first: &ChannelForcingInterval,
    shape: Shape,
) -> TestResult {
    let original = origin.snapshot_data()?;
    let saved = current.snapshot_data()?;
    let trusted = origin.restore(&current.snapshot())?;
    let cold = origin.import_snapshot(&saved)?;
    assert_eq!(trusted.snapshot_data()?, saved);
    assert_eq!(cold.snapshot_data()?, saved);
    let start = current.time();
    let end = start + 1.0 / 1024.0;
    let continuation = ChannelForcingInterval {
        start_s: start,
        end_s: end,
        ..first.clone()
    };
    let continued = current.step(end, &continuation)?.next;
    assert_eq!(continued.time().to_bits(), end.to_bits());
    check_mapping(&continued, &continuation, shape)?;
    let expected = continued.snapshot_data()?;
    assert_eq!(
        trusted.step(end, &continuation)?.next.snapshot_data()?,
        expected
    );
    assert_eq!(
        cold.step(end, &continuation)?.next.snapshot_data()?,
        expected
    );
    assert_eq!(current.snapshot_data()?, saved);
    assert_eq!(origin.snapshot_data()?, original);
    Ok(())
}

#[test]
fn packed_stage_and_discharge_roots_preserve_both_endpoint_orders_and_restart() -> TestResult {
    for (shape, outward_q) in [
        (
            Shape::Rectangle {
                gravity: 1.0,
                width: 1.0,
            },
            0.162,
        ),
        (
            Shape::Triangle {
                gravity: 8.0,
                coefficient: 1.0,
            },
            0.52488,
        ),
    ] {
        for stage_side in [Side::Left, Side::Right] {
            let origin = reach(
                definition(shape, 1, 128.0, 4.0, 1.0),
                initial(shape, &[128.0], &[1.0], &[0.0]),
            )?;
            let before = origin.snapshot_data()?;
            let forcing = frame(
                stage_side,
                ChannelBoundary::Stage { elevation_m: 1.1 },
                outward_q,
            );
            let current = origin.step(forcing.end_s, &forcing)?.next;
            assert_eq!(current.time().to_bits(), forcing.end_s.to_bits());
            check_mapping(&current, &forcing, shape)?;
            assert_eq!(origin.snapshot_data()?, before);
            let receipt = current
                .last_step()?
                .ok_or("Missing mixed-policy packed receipt")?;
            let discharge = receipt.boundary_traces[1 - stage_side.index()]
                .as_ref()
                .ok_or("Missing packed root")?;
            close(discharge.depth_m, 0.81);
            assert!(discharge.root_iterations > 0);
            assert!(!discharge.cap_compatibility_approximation);
            check_restart(&origin, current, &forcing, shape)?;
        }
    }
    Ok(())
}

#[test]
fn packed_cap_and_interior_discharge_roots_preserve_both_endpoint_orders_and_restart() -> TestResult
{
    for (shape, cap, cap_q, ordinary_q) in [
        (
            Shape::Rectangle {
                gravity: 1.0,
                width: 1.0,
            },
            2.0,
            -3.0_f64.sqrt(),
            0.162,
        ),
        (
            Shape::Triangle {
                gravity: 8.0,
                coefficient: 1.0,
            },
            1.5,
            -(9.0 / 4.0) * (95.0_f64 / 27.0).sqrt(),
            0.52488,
        ),
    ] {
        for cap_side in [Side::Left, Side::Right] {
            let origin = reach(
                definition(shape, 1, 128.0, cap, 1.0),
                initial(shape, &[128.0], &[1.0], &[0.0]),
            )?;
            let before = origin.snapshot_data()?;
            let forcing = frame(
                cap_side,
                ChannelBoundary::Discharge {
                    discharge_m3_per_s: cap_side.sign() * cap_q,
                },
                ordinary_q,
            );
            let current = origin.step(forcing.end_s, &forcing)?.next;
            assert_eq!(current.time().to_bits(), forcing.end_s.to_bits());
            check_mapping(&current, &forcing, shape)?;
            assert_eq!(origin.snapshot_data()?, before);
            let receipt = current
                .last_step()?
                .ok_or("Missing cap/root packed receipt")?;
            let cap_trace = receipt.boundary_traces[cap_side.index()]
                .as_ref()
                .ok_or("Missing mapped cap trace")?;
            let ordinary = receipt.boundary_traces[1 - cap_side.index()]
                .as_ref()
                .ok_or("Missing mapped interior root")?;
            assert!(cap_trace.depth_m <= cap);
            close(cap_trace.depth_m, cap);
            assert!(
                cap_trace.cap_compatibility_approximation,
                "Rounded irrational cap target must exercise certified uncertainty-band packing"
            );
            assert_eq!(cap_trace.depth_m.to_bits(), cap.to_bits());
            assert_eq!(cap_trace.root_iterations, 0);
            assert!(cap_trace.velocity_compatibility_abs_bound_m_per_s > 0.0);
            assert!(cap_trace.water_compatibility_abs_bound_m3_per_s > 0.0);
            assert!(cap_trace.momentum_compatibility_abs_bound_m4_per_s2 > 0.0);
            close(ordinary.depth_m, 0.81);
            assert!(ordinary.root_iterations > 0);
            assert!(!ordinary.cap_compatibility_approximation);
            check_restart(&origin, current, &forcing, shape)?;
        }
    }
    Ok(())
}

#[test]
fn packed_discharge_roots_use_each_rectangular_endpoint_width() -> TestResult {
    let shape = Shape::Rectangle {
        gravity: 1.0,
        width: 3.0,
    };
    let mut layout = definition(shape, 2, 8.0, 2.0, 1.0);
    let hydr::channel::ChannelGeometry::Rectangular { cells } = &mut layout.geometry else {
        return Err("Heterogeneous-width fixture requires rectangles".into());
    };
    cells[1].width_m = 7.0;
    let origin = reach(
        layout,
        hydr::channel::ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![24.0, 56.0],
            longitudinal_momentum_m4_per_s: vec![0.0, 0.0],
        },
    )?;
    let before = origin.snapshot_data()?;
    let dt = 1.0 / 1024.0;
    let targets = [-0.03_f64, 0.07_f64];
    let forcing = ChannelForcingInterval {
        start_s: 0.0,
        end_s: dt,
        left: ChannelBoundary::Discharge {
            discharge_m3_per_s: targets[0],
        },
        right: ChannelBoundary::Discharge {
            discharge_m3_per_s: targets[1],
        },
        lateral_inputs: Vec::new(),
    };
    let next = origin.step(dt, &forcing)?.next;
    assert_eq!(next.time().to_bits(), dt.to_bits());
    let receipt = next
        .last_step()?
        .ok_or("Missing heterogeneous packed receipt")?;
    let left = receipt.boundary_traces[0]
        .as_ref()
        .ok_or("Missing width-three trace")?;
    let right = receipt.boundary_traces[1]
        .as_ref()
        .ok_or("Missing width-seven trace")?;
    // Equal outward discharge per width gives the same independent rectangular wave relation.
    assert!((left.depth_m - right.depth_m).abs() <= 1e-12);
    assert!((left.outward_velocity_m_per_s - right.outward_velocity_m_per_s).abs() <= 1e-12);
    close(left.area_m2, 3.0 * left.depth_m);
    close(right.area_m2, 7.0 * right.depth_m);
    for (endpoint, (trace, width)) in [(left, 3.0), (right, 7.0)].into_iter().enumerate() {
        let endpoint_shape = Shape::Rectangle {
            gravity: 1.0,
            width,
        };
        check_trace(trace, next.config(), endpoint_shape, 1.0);
        assert!(trace.root_iterations > 0);
        assert!(!trace.cap_compatibility_approximation);
        assert_eq!(
            trace.outward_discharge_m3_per_s.to_bits(),
            targets[endpoint].abs().to_bits()
        );
        assert_eq!(
            receipt.face_volume_m3[endpoint * 2].to_bits(),
            (targets[endpoint] * dt).to_bits()
        );
    }
    let saved = next.snapshot_data()?;
    let retained = &saved
        .last_interval
        .as_ref()
        .ok_or("Missing heterogeneous saved interval")?
        .forcing;
    for (endpoint, boundary) in [&retained.left, &retained.right].into_iter().enumerate() {
        let ChannelBoundary::Discharge { discharge_m3_per_s } = boundary else {
            return Err("Heterogeneous fixture changes scalar policy".into());
        };
        assert_eq!(discharge_m3_per_s.to_bits(), targets[endpoint].to_bits());
    }
    assert_eq!(origin.snapshot_data()?, before);
    check_water(&next)
}
