//! Flux port endpoints: rest, uniform flow, water budgets and the CFL bound.
//!
//! Tolerances are frozen before any run. A test that misses one fails.

use super::common::{Reach, Shape, TestResult, check_water, definition, initial, reach};
use hydr::channel::{
    ChannelBoundary, ChannelError, ChannelFluxPort, ChannelForcingInterval,
    ChannelTriangularReconstruction,
};

/// Largest discharge in m³/s and stage change in m that rest may show after `INTERVALS`.
pub const REST_TOLERANCE: f64 = 1e-12;

/// Largest relative change of any depth or discharge under uniform flow after `INTERVALS`.
pub const UNIFORM_RELATIVE: f64 = 1e-12;

/// Largest relative difference between a CFL limit and its expected value.
pub const CFL_RELATIVE: f64 = 1e-14;

/// Accepted intervals per rest and uniform-flow run.
pub const INTERVALS: usize = 50;

/// Gravity in m/s² of every fixture.
const GRAVITY: f64 = 9.81;

/// Cell length in metres.
const LENGTH: f64 = 4.0;

/// Highest supported depth in metres.
const CAP: f64 = 10.0;

pub const SHAPES: [Shape; 2] = [
    Shape::Rectangle {
        gravity: GRAVITY,
        width: 2.0,
    },
    Shape::Triangle {
        gravity: GRAVITY,
        coefficient: 1.0,
    },
];

/// Riemann invariant `φ` of depth `h`: `2c` for the rectangle, `4c` for the triangle.
pub fn invariant(shape: Shape, h: f64) -> f64 {
    match shape {
        Shape::Rectangle { .. } => 2.0 * shape.sound(h),
        Shape::Triangle { .. } => 4.0 * shape.sound(h),
    }
}

/// A flux port with checked operands.
pub fn port(
    water: f64,
    momentum: f64,
    speed: f64,
    invariant: Option<f64>,
) -> Result<ChannelBoundary, ChannelError> {
    ChannelFluxPort::new(
        water,
        momentum,
        speed,
        invariant,
        "test provider of exact flux",
    )
    .map(ChannelBoundary::FluxPort)
}

/// Ports at both ends that supply the exact physical flux of a uniform state `h`, `u`.
///
/// The left port takes `-Q` as its outward rate and the right port `+Q`. Both supply the
/// same normal momentum flux `Q u + p`. Each invariant moves into the reach.
pub fn uniform_ports(shape: Shape, h: f64, u: f64) -> Result<[ChannelBoundary; 2], ChannelError> {
    let q = shape.area(h) * u;
    let momentum = q * u + shape.pressure(h);
    let speed = u.abs() + shape.sound(h);
    let phi = invariant(shape, h);
    Ok([
        port(-q, momentum, speed, Some(u + phi))?,
        port(q, momentum, speed, Some(-u + phi))?,
    ])
}

/// Reach of `shape` with `count` cells of depth `h` and velocity `u`.
pub fn uniform_reach(
    shape: Shape,
    count: usize,
    h: f64,
    u: f64,
) -> Result<Reach, Box<dyn std::error::Error>> {
    reach(
        definition(shape, count, LENGTH, CAP, 1.0),
        initial(
            shape,
            &vec![LENGTH; count],
            &vec![h; count],
            &vec![u; count],
        ),
    )
}

/// One cell of depth `h` at rest, with first-order triangular faces.
///
/// A limited-linear triangle splits each cell into half-cells. One port speed then crosses a
/// half-cell, which halves the CFL limit. The first-order face keeps one full cell.
fn first_order_cell(shape: Shape, h: f64) -> Result<Reach, Box<dyn std::error::Error>> {
    let mut layout = definition(shape, 1, LENGTH, CAP, 1.0);
    layout.config.triangular_reconstruction = ChannelTriangularReconstruction::FirstOrderReference;
    reach(layout, initial(shape, &[LENGTH], &[h], &[0.0]))
}

/// Held `left` and `right` from the accepted time to `end`.
pub fn held(
    reach: &Reach,
    end: f64,
    left: &ChannelBoundary,
    right: &ChannelBoundary,
) -> ChannelForcingInterval {
    ChannelForcingInterval {
        start_s: reach.time(),
        end_s: end,
        left: left.clone(),
        right: right.clone(),
        lateral_inputs: Vec::new(),
    }
}

/// Accepts `count` intervals of at most `span` seconds each.
pub fn advance(
    mut current: Reach,
    count: usize,
    span: f64,
    left: &ChannelBoundary,
    right: &ChannelBoundary,
) -> Result<Reach, Box<dyn std::error::Error>> {
    for _ in 0..count {
        let end = current.time() + span;
        current = current.step(end, &held(&current, end, left, right))?.next;
        check_water(&current)?;
    }
    Ok(current)
}

#[test]
fn lake_at_rest_with_a_zero_flux_port_matches_the_wall() -> TestResult {
    for shape in SHAPES {
        let h = 1.0;
        let closed = port(
            0.0,
            shape.pressure(h),
            shape.sound(h),
            Some(invariant(shape, h)),
        )?;
        let wall = ChannelBoundary::Wall;
        for (left, right) in [(&closed, &wall), (&wall, &closed), (&closed, &closed)] {
            let origin = uniform_reach(shape, 4, h, 0.0)?;
            let ported = advance(origin.clone(), INTERVALS, 0.25, left, right)?;
            let walled = advance(origin, INTERVALS, 0.25, &wall, &wall)?;
            for (port_cell, wall_cell) in ported.state()?.iter().zip(walled.state()?) {
                let stage = port_cell.stage_m.ok_or("rest cell dried")?;
                assert!(
                    port_cell.discharge_m3_per_s.abs() <= REST_TOLERANCE,
                    "{shape:?}: {port_cell:?}"
                );
                assert!(
                    (stage - h).abs() <= REST_TOLERANCE,
                    "{shape:?}: {port_cell:?}"
                );
                let wall_stage = wall_cell.stage_m.ok_or("walled rest cell dried")?;
                assert!(
                    (stage - wall_stage).abs() <= REST_TOLERANCE,
                    "{shape:?}: {port_cell:?} vs {wall_cell:?}"
                );
                assert!(
                    (port_cell.discharge_m3_per_s - wall_cell.discharge_m3_per_s).abs()
                        <= REST_TOLERANCE
                );
            }
            let receipt = ported.last_step()?.ok_or("missing rest receipt")?;
            assert_eq!(receipt.boundary_traces, [None, None]);
            let ledger = ported.ledger();
            assert_eq!(ledger.boundary_in_m3, 0.0);
            assert_eq!(ledger.boundary_out_m3, 0.0);
        }
    }
    Ok(())
}

#[test]
fn exact_uniform_flux_keeps_uniform_flow() -> TestResult {
    for shape in SHAPES {
        let h = 1.0;
        let c = shape.sound(h);
        // Subcritical Fr 0.5 and supercritical Fr 2, in both directions.
        for u in [0.5 * c, -0.5 * c, 2.0 * c, -2.0 * c] {
            let [left, right] = uniform_ports(shape, h, u)?;
            let origin = uniform_reach(shape, 4, h, u)?;
            let q = shape.area(h) * u;
            let accepted = advance(origin, INTERVALS, 0.25, &left, &right)?;
            for cell in accepted.state()? {
                let relative_depth = (cell.depth_m - h).abs() / h;
                let relative_discharge = (cell.discharge_m3_per_s - q).abs() / q.abs();
                assert!(
                    relative_depth <= UNIFORM_RELATIVE && relative_discharge <= UNIFORM_RELATIVE,
                    "{shape:?} u={u}: {cell:?}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn port_water_enters_the_ledger_exactly() -> TestResult {
    for shape in SHAPES {
        let h = 1.0;
        let (inflow, outflow) = (0.25, 0.125);
        let pressure = shape.pressure(h);
        let speed = shape.sound(h) + 1.0;
        let left = port(-inflow, pressure, speed, Some(invariant(shape, h)))?;
        let right = port(outflow, pressure, speed, Some(invariant(shape, h)))?;
        let mut current = uniform_reach(shape, 4, h, 0.0)?;
        let initial_storage = current.ledger().storage_m3;
        let (mut total_in, mut total_out) = (0.0, 0.0);
        for _ in 0..10 {
            let end = current.time() + 0.25;
            current = current.step(end, &held(&current, end, &left, &right))?.next;
            check_water(&current)?;
            let receipt = current.last_step()?.ok_or("missing budget receipt")?;
            let dt = receipt.end_s - receipt.start_s;
            let entering = receipt.face_volume_m3[0];
            let leaving = receipt.face_volume_m3[4];
            assert_eq!(entering.to_bits(), (inflow * dt).to_bits(), "{shape:?}");
            assert_eq!(leaving.to_bits(), (outflow * dt).to_bits(), "{shape:?}");
            assert_eq!(receipt.boundary_in_m3.to_bits(), entering.to_bits());
            assert_eq!(receipt.boundary_out_m3.to_bits(), leaving.to_bits());
            total_in += entering;
            total_out += leaving;
        }
        let ledger = current.ledger();
        let scale = ledger.storage_m3.max(1.0);
        assert!((ledger.boundary_in_m3 - total_in).abs() <= 1e-15 * scale);
        assert!((ledger.boundary_out_m3 - total_out).abs() <= 1e-15 * scale);
        let net = ledger.storage_m3 - initial_storage - (total_in - total_out);
        assert!(net.abs() <= 1e-12 * scale, "{shape:?}: residual={net}");
    }
    Ok(())
}

#[test]
fn port_momentum_enters_the_end_cells_with_zero_side_pressure() -> TestResult {
    for shape in SHAPES {
        let h = 1.0;
        let [left, right] = uniform_ports(shape, h, 0.5 * shape.sound(h))?;
        let ChannelBoundary::FluxPort(port) = &left else {
            return Err("uniform port is not a FluxPort".into());
        };
        let rate = port.outward_momentum_rate_m4_per_s2();
        let origin = uniform_reach(shape, 4, h, 0.5 * shape.sound(h))?;
        let next = advance(origin, 1, 0.25, &left, &right)?;
        let receipt = next.last_step()?.ok_or("missing momentum receipt")?;
        let dt = receipt.end_s - receipt.start_s;
        // Both faces carry the normal momentum flux along the increasing coordinate.
        assert_eq!(
            receipt.face_momentum_m4_per_s[0].to_bits(),
            (rate * dt).to_bits()
        );
        assert_eq!(
            receipt.face_momentum_m4_per_s[4].to_bits(),
            (rate * dt).to_bits()
        );
        assert_eq!(receipt.right_pressure_impulse_m4_per_s[0], 0.0);
        assert_eq!(receipt.left_pressure_impulse_m4_per_s[4], 0.0);
        // The left end cell gains `+R dt` and the right end cell `-R dt`: `-σ R dt`.
        let endpoints = next.coupling_endpoints()?;
        let impulse = endpoints.momentum_impulse_m4_per_s.try_to_vec::<f64>()?;
        assert_eq!(impulse[0].to_bits(), (rate * dt).to_bits());
        assert_eq!(impulse[1].to_bits(), (-(rate * dt)).to_bits());
    }
    Ok(())
}

#[test]
fn port_speed_bounds_the_end_cells_in_both_directions() -> TestResult {
    for shape in SHAPES {
        let h = 1.0;
        let speed = 10.0;
        let closed = port(0.0, shape.pressure(h), speed, Some(invariant(shape, h)))?;
        let origin = first_order_cell(shape, h)?;
        let next = advance(origin.clone(), 1, 1e-3, &closed, &closed)?;
        let receipt = next.last_step()?.ok_or("missing CFL receipt")?;
        let cfl = origin.config().cfl;
        // One cell between two ports: the inward speeds sum to `2 S` over `2 L`.
        let expected = cfl * LENGTH / speed;
        let limit = receipt.accepted_start_cfl_limit_s;
        assert!(
            (limit - expected).abs() <= CFL_RELATIVE * expected,
            "{shape:?}: limit={limit}, expected={expected}"
        );
        assert!(limit < cfl * LENGTH / shape.sound(h), "{shape:?}");
    }
    Ok(())
}

#[test]
fn dry_end_cell_takes_the_exterior_invariant() -> TestResult {
    for shape in SHAPES {
        let (speed, phi) = (1.0, 7.0);
        let filling = port(-0.01, 0.0, speed, Some(phi))?;
        let origin = first_order_cell(shape, 0.0)?;
        let next = advance(origin.clone(), 1, 1e-3, &filling, &ChannelBoundary::Wall)?;
        let receipt = next.last_step()?.ok_or("missing dry-front receipt")?;
        // The dry cell sees the larger of the invariant and the speed, from one side only.
        let expected = origin.config().cfl * 2.0 * LENGTH / phi;
        let limit = receipt.accepted_start_cfl_limit_s;
        assert!(
            (limit - expected).abs() <= CFL_RELATIVE * expected,
            "{shape:?}: limit={limit}, expected={expected}"
        );
        assert!(next.state()?[0].volume_m3 > 0.0);
    }
    Ok(())
}
