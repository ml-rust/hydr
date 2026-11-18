//! Lake at rest through a port, flat and with a bed step up and down, on one face and four.

use crate::fixture::{
    Engine, Grid, TestResult, complete, engine, ghosts, host, prepare, spec, x_face,
};
use hydr::model::CellState;
use hydr_solver::internal::PortReachEnd;

/// Frozen rest criteria from the port plan, in m²/s and m.
const DISCHARGE_LIMIT: f64 = 1e-12;
const STAGE_LIMIT: f64 = 1e-12;
const INTERVALS: usize = 50;

const STAGE: f64 = 1.0;
const SURFACE_BED: f64 = 0.2;

/// A resting 3-column grid of `rows` rows at stage [`STAGE`] over [`SURFACE_BED`].
fn lake(rows: usize) -> TestResult<Engine> {
    engine(Grid::uniform(
        3,
        rows,
        [1.0, 1.5],
        SURFACE_BED,
        CellState {
            h: STAGE - SURFACE_BED,
            qx: 0.0,
            qy: 0.0,
        },
    ))
}

/// Steps the lake with a west port on every row and checks rest after each interval.
fn check_rest(rows: usize, ghost_bed: f64, ghost_depth: f64) -> TestResult {
    let mut solver = lake(rows)?;
    let faces = (0..rows).map(|row| x_face(0, row)).collect();
    let ports = solver.coupling_ports(&[spec(faces, ghost_bed, PortReachEnd::Right)])?;
    let ghost = ghosts(&solver, &ports, &[ghost_depth], &[0.0])?;
    let start = solver.state()?;
    for _ in 0..INTERVALS {
        let stage = prepare(&solver, 0.5, Some((&ports, &ghost)))?;
        let fluxes = stage
            .port_fluxes()
            .ok_or("a ported stage reports port fluxes")?;
        for water in host(&fluxes.face_water_into_surface_m2_per_s)? {
            assert!(water.abs() <= DISCHARGE_LIMIT, "port water flux {water}");
        }
        solver = complete(&solver, &stage)?;
        for (cell, (now, then)) in solver
            .state()?
            .cells()
            .iter()
            .zip(start.cells())
            .enumerate()
        {
            assert!(
                now.qx.abs() <= DISCHARGE_LIMIT && now.qy.abs() <= DISCHARGE_LIMIT,
                "cell={cell}, q=[{}, {}]",
                now.qx,
                now.qy
            );
            assert!(
                (now.h - then.h).abs() <= STAGE_LIMIT,
                "cell={cell}, stage change={}",
                now.h - then.h
            );
        }
    }
    Ok(())
}

#[test]
fn flat_port_keeps_rest() -> TestResult {
    check_rest(1, SURFACE_BED, STAGE - SURFACE_BED)?;
    check_rest(4, SURFACE_BED, STAGE - SURFACE_BED)
}

#[test]
fn step_down_into_the_reach_keeps_rest() -> TestResult {
    let bed = -0.3;
    check_rest(1, bed, STAGE - bed)?;
    check_rest(4, bed, STAGE - bed)
}

#[test]
fn step_up_into_the_reach_keeps_rest() -> TestResult {
    let bed = 0.6;
    check_rest(1, bed, STAGE - bed)?;
    check_rest(4, bed, STAGE - bed)
}

/// Water below the higher bed on both sides moves no volume.
#[test]
fn water_below_the_higher_bed_stays_put() -> TestResult {
    let solver = lake(2)?;
    let ports = solver.coupling_ports(&[spec(
        vec![x_face(0, 0), x_face(0, 1)],
        STAGE + 0.5,
        PortReachEnd::Right,
    )])?;
    let ghost = ghosts(&solver, &ports, &[0.0], &[0.0])?;
    let stage = prepare(&solver, 0.5, Some((&ports, &ghost)))?;
    let fluxes = stage
        .port_fluxes()
        .ok_or("a ported stage reports port fluxes")?;
    for water in host(&fluxes.face_water_into_surface_m2_per_s)? {
        assert_eq!(water, 0.0);
    }
    let next = complete(&solver, &stage)?;
    let step = next.last_step()?.ok_or("an accepted step has a receipt")?;
    let port_faces = [x_face(0, 0), x_face(0, 1)];
    for face in step
        .faces
        .iter()
        .filter(|face| port_faces.contains(&face.face))
    {
        assert_eq!(face.volume, 0.0, "{face:?}");
    }
    assert_eq!(step.exchange_in_volume, 0.0);
    assert_eq!(step.exchange_out_volume, 0.0);
    Ok(())
}
