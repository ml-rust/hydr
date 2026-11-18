//! Port face speeds bound the stage CFL limit. Port outflow bounds the depletion limit.

use crate::fixture::{
    Engine, GRAVITY, Grid, TestResult, engine, ghosts, host, prepare, spec, x_face,
};
use hydr::model::CellState;
use hydr_solver::internal::PortReachEnd;

fn resting(h: f64, bed: f64, spacing: [f64; 2]) -> TestResult<Engine> {
    engine(Grid::uniform(
        1,
        1,
        spacing,
        bed,
        CellState {
            h,
            qx: 0.0,
            qy: 0.0,
        },
    ))
}

/// A fast ghost entering a resting cell binds the CFL limit through the port face.
#[test]
fn port_speed_enters_the_cfl_limit() -> TestResult {
    let (dx, h) = (2.0, 0.5);
    let solver = resting(h, 0.0, [dx, 3.0])?;
    let portless = prepare(&solver, 1.0e-3, None)?;
    assert!(portless.port_fluxes().is_none());
    let ports = solver.coupling_ports(&[spec(vec![x_face(0, 0)], 0.0, PortReachEnd::Right)])?;
    let ghost_velocity = 10.0;
    let ghost = ghosts(&solver, &ports, &[h], &[ghost_velocity * h])?;
    let stage = prepare(&solver, 1.0e-3, Some((&ports, &ghost)))?;
    let fluxes = stage
        .port_fluxes()
        .ok_or("a ported stage reports port fluxes")?;
    let speed = host(&fluxes.face_signal_speed_m_per_s)?[0];
    assert_eq!(
        host(&fluxes.port_signal_speed_m_per_s)?[0].to_bits(),
        speed.to_bits()
    );
    // The port speed is the HLL signal magnitude, faster than the resting surface wave.
    let lower = host(&fluxes.face_lower_wave_into_surface_m_per_s)?[0];
    let upper = host(&fluxes.face_upper_wave_into_surface_m_per_s)?[0];
    assert_eq!(speed.to_bits(), lower.abs().max(upper.abs()).to_bits());
    assert!(speed > (GRAVITY * h).sqrt());
    let cfl = solver.config().cfl;
    assert!(stage.cfl_limit() <= cfl / (speed / dx));
    assert!(stage.cfl_limit() < portless.cfl_limit());
    Ok(())
}

/// A resting cell draining onto a dry ghost below it exports through the port face only.
#[test]
fn port_outflow_enters_the_depletion_limit() -> TestResult {
    let (dx, dy, h) = (2.0, 3.0, 0.01);
    let solver = resting(h, 0.0, [dx, dy])?;
    let portless = prepare(&solver, 1.0e-3, None)?;
    assert_eq!(portless.depletion_limit(), f64::INFINITY);
    let ports = solver.coupling_ports(&[spec(vec![x_face(0, 0)], -1.0, PortReachEnd::Right)])?;
    let ghost = ghosts(&solver, &ports, &[0.0], &[0.0])?;
    let stage = prepare(&solver, 1.0e-3, Some((&ports, &ghost)))?;
    let fluxes = stage
        .port_fluxes()
        .ok_or("a ported stage reports port fluxes")?;
    let water = host(&fluxes.face_water_into_surface_m2_per_s)?[0];
    assert!(
        water < 0.0,
        "a wet cell beside a dry lower ghost drains, water={water}"
    );
    let outflow = -(water * dy);
    assert_eq!(fluxes.outflow_rate_m3_per_s.to_bits(), outflow.to_bits());
    assert_eq!(fluxes.inflow_rate_m3_per_s, 0.0);
    assert_eq!(
        stage.depletion_limit().to_bits(),
        ((h * dx * dy) / outflow).to_bits()
    );
    // Post-transport water is net of the port outflow.
    let available = host(stage.available_volume_m3())?[0];
    let expected = h + (water * dy / (dx * dy)) * stage.dt();
    assert!((available - expected * dx * dy).abs() <= 1e-15 * h * dx * dy);
    Ok(())
}
