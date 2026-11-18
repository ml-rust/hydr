//! Port water in the exchange totals, and port-aware coordinator restore.

use crate::fixture::{
    Engine, Grid, Runtime, TestResult, complete, engine, ghosts, host, prepare, rejection,
    relative, spec, x_face,
};
use hydr::model::CellState;
use hydr::solver::Solver;
use hydr_solver::internal::{PortGhosts, PortReachEnd, SurfacePorts};

/// Frozen relative tolerance between the summed face receipts and the exchange total.
const FACE_SUM_RELATIVE: f64 = 1e-14;

/// One column of two cells, 2 m by 1.5 m, with a two-face west port.
fn column(h: f64) -> TestResult<Engine> {
    engine(Grid::uniform(
        1,
        2,
        [2.0, 1.5],
        0.0,
        CellState {
            h,
            qx: 0.0,
            qy: 0.0,
        },
    ))
}

fn west_port(
    solver: &Engine,
    bed: f64,
    depth: f64,
) -> TestResult<(SurfacePorts<Runtime>, PortGhosts<Runtime>)> {
    let ports = solver.coupling_ports(&[spec(
        vec![x_face(0, 0), x_face(0, 1)],
        bed,
        PortReachEnd::Right,
    )])?;
    let ghost = ghosts(solver, &ports, &[depth], &[0.0])?;
    Ok((ports, ghost))
}

/// Sums port face receipt volumes into the surface. The ghost is the left side.
fn face_volume(solver: &Engine) -> TestResult<[f64; 2]> {
    let step = solver
        .last_step()?
        .ok_or("an accepted step has a receipt")?;
    let mut totals = [0.0_f64; 2];
    for face in step
        .faces
        .iter()
        .filter(|face| [x_face(0, 0), x_face(0, 1)].contains(&face.face))
    {
        totals[0] += face.volume.max(0.0);
        totals[1] += (-face.volume).max(0.0);
    }
    Ok(totals)
}

#[test]
fn port_outflow_counts_as_exchange_out() -> TestResult {
    let solver = column(0.02)?;
    let (ports, ghost) = west_port(&solver, -1.0, 0.0)?;
    let stage = prepare(&solver, 0.5, Some((&ports, &ghost)))?;
    let fluxes = stage
        .port_fluxes()
        .ok_or("a ported stage reports port fluxes")?;
    let outflow = fluxes.outflow_rate_m3_per_s;
    assert!(outflow > 0.0);
    assert_eq!(fluxes.inflow_rate_m3_per_s, 0.0);
    let reach = host(&fluxes.reach_outward_water_rate_m3_per_s)?[0];
    assert!(relative(reach, -outflow) <= FACE_SUM_RELATIVE);
    let next = complete(&solver, &stage)?;
    let step = next.last_step()?.ok_or("an accepted step has a receipt")?;
    assert_eq!(
        step.exchange_out_volume.to_bits(),
        (outflow * stage.dt()).to_bits()
    );
    assert_eq!(step.exchange_in_volume, 0.0);
    assert_eq!(step.boundary_in, 0.0);
    assert_eq!(step.boundary_out, 0.0);
    assert_eq!(
        next.ledger().exchange_out_m3.to_bits(),
        step.exchange_out_volume.to_bits()
    );
    let [into, out_of] = face_volume(&next)?;
    assert_eq!(into, 0.0);
    assert!(relative(out_of, step.exchange_out_volume) <= FACE_SUM_RELATIVE);
    Ok(())
}

#[test]
fn port_inflow_counts_as_exchange_in() -> TestResult {
    let solver = column(0.1)?;
    let (ports, ghost) = west_port(&solver, 0.0, 0.5)?;
    let stage = prepare(&solver, 0.5, Some((&ports, &ghost)))?;
    let fluxes = stage
        .port_fluxes()
        .ok_or("a ported stage reports port fluxes")?;
    let inflow = fluxes.inflow_rate_m3_per_s;
    assert!(inflow > 0.0);
    assert_eq!(fluxes.outflow_rate_m3_per_s, 0.0);
    let next = complete(&solver, &stage)?;
    let step = next.last_step()?.ok_or("an accepted step has a receipt")?;
    assert_eq!(
        step.exchange_in_volume.to_bits(),
        (inflow * stage.dt()).to_bits()
    );
    assert_eq!(step.exchange_out_volume, 0.0);
    let [into, out_of] = face_volume(&next)?;
    assert_eq!(out_of, 0.0);
    assert!(relative(into, step.exchange_in_volume) <= FACE_SUM_RELATIVE);
    let storage = step.storage_after - step.storage_before;
    assert!((storage - step.exchange_in_volume).abs() <= solver.config().volume_absolute);
    Ok(())
}

#[test]
fn port_water_restores_only_with_its_ports() -> TestResult {
    let solver = column(0.02)?;
    let (ports, ghost) = west_port(&solver, -1.0, 0.0)?;
    let stage = prepare(&solver, 0.5, Some((&ports, &ghost)))?;
    let next = complete(&solver, &stage)?;
    let data = next.snapshot().data()?;
    let message = rejection(column(0.02)?.restore_data(&data));
    assert!(
        message.contains("restore this package through the surface-channel coordinator"),
        "{message}"
    );
    let message = rejection(column(0.02)?.coupling_restore_data(&data, None));
    assert!(message.contains("source exchange volume"), "{message}");
    let mut restored = column(0.02)?;
    let own = restored.coupling_ports(&[spec(
        vec![x_face(0, 0), x_face(0, 1)],
        -1.0,
        PortReachEnd::Right,
    )])?;
    restored.coupling_restore_data(&data, Some(&own))?;
    assert_eq!(restored.snapshot().data()?, data);
    let resumed = Solver::<Runtime>::coupling_resume(
        crate::runtime::client(),
        &data,
        data.setup.forcing.clone(),
        Some(&ports),
    )?;
    assert_eq!(resumed.snapshot().data()?, data);
    // A port set that misses one port face leaves part of the port water unexplained.
    let partial = solver.coupling_ports(&[spec(vec![x_face(0, 0)], -1.0, PortReachEnd::Right)])?;
    let message = rejection(column(0.02)?.coupling_restore_data(&data, Some(&partial)));
    assert!(message.contains("source exchange volume"), "{message}");
    // Ports of another grid reject before import.
    let other = engine(Grid::uniform(
        3,
        1,
        [1.0, 1.0],
        0.0,
        CellState {
            h: 0.1,
            qx: 0.0,
            qy: 0.0,
        },
    ))?;
    let foreign = other.coupling_ports(&[spec(vec![x_face(0, 0)], 0.0, PortReachEnd::Right)])?;
    let message = rejection(column(0.02)?.coupling_restore_data(&data, Some(&foreign)));
    assert!(
        message.contains("build the ports from this engine"),
        "{message}"
    );
    Ok(())
}
