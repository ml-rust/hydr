use super::common::{
    Policy, Shape, Side, TestResult, check_water, definition, initial, interval, reach,
};
use crate::reference::{
    Geometry, left_cell_average, mean_errors, right_cell_average, trace, uniform_initial_signal,
};

fn measure(
    shape: Shape,
    geometry: Geometry,
    depth: f64,
    policy: Policy,
    side: Side,
    count: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let dx = 20.0 / count as f64;
    // The CFL limit sets every step, so time refines with space.
    let mut current = reach(
        definition(shape, count, dx, 4.0, 1.0),
        initial(
            shape,
            &vec![dx; count],
            &vec![1.0; count],
            &vec![0.0; count],
        ),
    )?;
    let outward_q = trace(geometry, depth).outward_discharge;
    let boundary = policy.boundary(side, depth, outward_q);
    let mut steps = 0;
    while current.time() < 2.0 {
        let start = current.time();
        current = current
            .step(2.0, &interval(side, boundary.clone(), start, 2.0))?
            .next;
        assert!(current.time() > start && current.time() <= 2.0);
        check_water(&current)?;
        steps += 1;
        assert!(steps < 1_000_000, "Scalar endpoint propagation stalls");
    }
    assert_eq!(current.time(), 2.0);
    let state = current.state()?;
    let errors = mean_errors(
        geometry,
        state.iter().enumerate().map(|(index, cell)| {
            let (left, right) = match side {
                Side::Right => {
                    let x = -20.0 + index as f64 * dx;
                    (x, x + dx)
                }
                Side::Left => {
                    let x = index as f64 * dx;
                    (x, x + dx)
                }
            };
            let exact = match side {
                Side::Right => right_cell_average(geometry, depth, left, right, 2.0),
                Side::Left => left_cell_average(geometry, depth, left, right, 2.0),
            };
            (
                dx,
                cell.volume_m3 / dx,
                cell.longitudinal_momentum_m4_per_s / dx,
                exact.0,
                exact.1,
            )
        }),
    )
    .ok_or("Missing endpoint propagation norm")?;
    println!(
        "scalar boundary {geometry:?} {policy:?} {side:?}: H={depth}, cells={count}, mean_abs_h={}, mean_abs_Q={}",
        errors.0, errors.1
    );
    assert!(errors.0 < 0.02, "Frozen depth ceiling: {errors:?}");
    assert!(errors.1 < 0.04, "Frozen discharge ceiling: {errors:?}");
    let receipt = current.last_step()?.ok_or("Missing propagation receipt")?;
    assert!(receipt.boundary_traces[side.index()].is_some());
    assert!(receipt.boundary_traces[1 - side.index()].is_none());
    Ok(errors)
}

/// Refinement meshes over the 20 m reach.
const COUNTS: [usize; 3] = [25, 50, 100];

fn check_refinement(policies: impl Fn(Shape, Geometry, f64) -> Vec<Policy>) -> TestResult {
    for (shape, geometry) in [
        (
            Shape::Rectangle {
                gravity: 1.0,
                width: 3.0,
            },
            Geometry::Rectangular,
        ),
        (
            Shape::Triangle {
                gravity: 2.0,
                coefficient: 1.0,
            },
            Geometry::Triangular,
        ),
    ] {
        for depth in [0.81, 1.21] {
            let signal = uniform_initial_signal(geometry, depth, 200, 2.0)
                .ok_or("Missing physical signal")?;
            assert!(signal.0.is_finite() && signal.0 > 0.0);
            assert!(signal.1.is_finite() && signal.1 > 0.0);
            for policy in policies(shape, geometry, depth) {
                let mut errors = Vec::new();
                for count in COUNTS {
                    errors.push(measure(shape, geometry, depth, policy, Side::Left, count)?);
                }
                // The right endpoint is the mirror image, so it must match the left errors.
                let mirrored = measure(shape, geometry, depth, policy, Side::Right, COUNTS[0])?;
                for (right, left) in [(mirrored.0, errors[0].0), (mirrored.1, errors[0].1)] {
                    assert!(
                        (right - left).abs() <= 1e-12 * left.abs(),
                        "Mirrored endpoint errors differ: right={right}, left={left}"
                    );
                }
                assert!(
                    errors.windows(2).all(|pair| pair[1].0 < pair[0].0),
                    "Depth refinement: {errors:?}"
                );
                assert!(
                    errors.windows(2).all(|pair| pair[1].1 < pair[0].1),
                    "Discharge refinement: {errors:?}"
                );
                let finest = errors.last().ok_or("Missing finest errors")?;
                assert!(
                    finest.0 < 0.25 * signal.0,
                    "Depth error={}, physical signal={}",
                    finest.0,
                    signal.0
                );
                assert!(
                    finest.1 < 0.25 * signal.1,
                    "Discharge error={}, physical signal={}",
                    finest.1,
                    signal.1
                );
            }
        }
    }
    Ok(())
}

#[test]
fn stage_propagates_exact_incoming_waves_under_refinement() -> TestResult {
    check_refinement(|_, _, _| vec![Policy::Stage])
}

#[test]
fn discharge_propagates_exact_incoming_waves_under_refinement() -> TestResult {
    check_refinement(|_, _, _| vec![Policy::Discharge])
}

#[test]
fn reservoir_propagates_exact_incoming_waves_under_refinement() -> TestResult {
    check_refinement(|shape, geometry, depth| {
        let entrance_loss_coefficient = 1.0;
        let incoming_speed = (-trace(geometry, depth).outward_velocity).max(0.0);
        let elevation_m = depth
            + (1.0 + entrance_loss_coefficient) * incoming_speed * incoming_speed
                / (2.0 * shape.gravity());
        vec![Policy::Reservoir {
            elevation_m,
            entrance_loss_coefficient,
        }]
    })
}
