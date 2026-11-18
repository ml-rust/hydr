//! Port face fluxes: upwind selection, interior-face identity, and momentum receipts.

use crate::fixture::{
    GRAVITY, Grid, TestResult, complete, engine, ghosts, host, prepare, relative, spec, x_face,
    y_face,
};
use hydr::model::CellState;
use hydr_solver::internal::PortReachEnd;

/// Frozen relative tolerance for momentum rounding against the host formula.
const MOMENTUM_RELATIVE: f64 = 1e-14;

/// A Froude 2 surface cell drains east into a faster ghost. The flux is the surface's own.
#[test]
fn supercritical_outflow_equals_upwind_physical_flux() -> TestResult {
    supercritical_outflow(true)
}

/// The same outflow rotated onto a north face: the y face takes its normal momentum.
#[test]
fn supercritical_outflow_through_a_y_face_equals_upwind_physical_flux() -> TestResult {
    supercritical_outflow(false)
}

/// Outflow through the east face of one cell, or through its north face when `!x_axis`.
fn supercritical_outflow(x_axis: bool) -> TestResult {
    let h = 1.0;
    let q = 2.0 * (GRAVITY * h).sqrt() * h;
    let tangent = 0.1;
    // The face is 3 m wide. Its normal discharge is `q`, its tangential discharge `tangent`.
    let (spacing, qx, qy, face) = if x_axis {
        ([2.0, 3.0], q, tangent, x_face(1, 0))
    } else {
        ([3.0, 2.0], tangent, q, y_face(0, 1))
    };
    let solver = engine(Grid::uniform(1, 1, spacing, 0.0, CellState { h, qx, qy }))?;
    // The ghost is on the right. A left reach end maps reach-positive onto the face axis.
    let ports = solver.coupling_ports(&[spec(vec![face], 0.0, PortReachEnd::Left)])?;
    let ghost_q = 3.0 * (GRAVITY * h).sqrt() * h;
    let ghost = ghosts(&solver, &ports, &[h], &[ghost_q])?;
    assert_eq!(host(ghost.face_normal_discharge_m2_per_s())?, vec![ghost_q]);
    let stage = prepare(&solver, 0.01, Some((&ports, &ghost)))?;
    let fluxes = stage
        .port_fluxes()
        .ok_or("a ported stage reports port fluxes")?;
    let velocity = q / h;
    let normal = q * velocity + 0.5 * GRAVITY * h * h;
    // Water leaves the surface, so the flux into the surface is the negated upwind discharge.
    assert_eq!(
        host(&fluxes.face_water_into_surface_m2_per_s)?[0].to_bits(),
        (-q).to_bits()
    );
    let actual = host(&fluxes.face_normal_momentum_m3_per_s2)?[0];
    assert!(
        relative(actual, normal) <= MOMENTUM_RELATIVE,
        "normal={actual}, upwind={normal}"
    );
    let actual = host(&fluxes.face_tangential_momentum_into_surface_m3_per_s2)?[0];
    assert!(
        relative(actual, -tangent * velocity) <= MOMENTUM_RELATIVE,
        "tangent={actual}, upwind={}",
        -tangent * velocity
    );
    assert_eq!(host(&fluxes.face_surface_pressure_m3_per_s2)?, vec![0.0]);
    // Every wave leaves the surface, so both bounds point into the reach.
    assert!(host(&fluxes.face_upper_wave_into_surface_m_per_s)?[0] <= 0.0);
    assert!(host(&fluxes.face_lower_wave_into_surface_m_per_s)?[0] <= 0.0);
    // The reach receives the same water and momentum along its outward normal.
    let width = 3.0;
    assert_eq!(
        host(&fluxes.reach_outward_water_rate_m3_per_s)?[0].to_bits(),
        (-q * width).to_bits()
    );
    let reach = host(&fluxes.reach_outward_momentum_rate_m4_per_s2)?[0];
    assert!(relative(reach, normal * width) <= MOMENTUM_RELATIVE);
    // Without a bed step the only junction reaction is the tangential momentum that leaves.
    let reaction = [
        host(&fluxes.junction_reaction_rate_m4_per_s2[0])?[0],
        host(&fluxes.junction_reaction_rate_m4_per_s2[1])?[0],
    ];
    let (normal_axis, tangent_axis) = if x_axis { (0, 1) } else { (1, 0) };
    assert_eq!(reaction[normal_axis], 0.0);
    assert!(relative(reaction[tangent_axis], tangent * velocity * width) <= MOMENTUM_RELATIVE);
    Ok(())
}

/// A ghost equal to the interior state gives the interior face flux bit for bit.
#[test]
fn uniform_ghost_equals_an_interior_face() -> TestResult {
    let cell = CellState {
        h: 0.8,
        qx: 0.35,
        qy: 0.0,
    };
    let solver = engine(Grid::uniform(3, 1, [1.0, 1.0], 0.25, cell))?;
    let ports = solver.coupling_ports(&[spec(vec![x_face(0, 0)], 0.25, PortReachEnd::Right)])?;
    let ghost = ghosts(&solver, &ports, &[cell.h], &[cell.qx])?;
    let stage = prepare(&solver, 0.01, Some((&ports, &ghost)))?;
    let next = complete(&solver, &stage)?;
    let step = next.last_step()?.ok_or("an accepted step has a receipt")?;
    let face = |target| {
        step.faces
            .iter()
            .find(|face| face.face == target)
            .ok_or("missing face receipt")
    };
    let port = face(x_face(0, 0))?;
    let interior = face(x_face(1, 0))?;
    let bits = |values: &[f64]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(&port.flux), bits(&interior.flux));
    assert_eq!(
        bits(&port.pressure_corrections),
        bits(&interior.pressure_corrections)
    );
    assert_eq!(port.signal_speed.to_bits(), interior.signal_speed.to_bits());
    assert_eq!(port.volume.to_bits(), interior.volume.to_bits());
    Ok(())
}
