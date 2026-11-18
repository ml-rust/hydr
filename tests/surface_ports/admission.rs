//! Port face admission: each rejection names the face and the next action.

use crate::fixture::{
    Grid, TestResult, engine, engine_with, ghosts, prepare, rejection, spec, x_face, y_face,
};
use hydr::model::CellState;
use hydr::solver::FluxPort;
use hydr_solver::internal::PortReachEnd;

const REST: CellState = CellState {
    h: 0.5,
    qx: 0.0,
    qy: 0.0,
};

/// A 3 by 2 grid with cell 4 inactive.
fn holed() -> Grid {
    let mut grid = Grid::uniform(3, 2, [2.0, 3.0], 0.0, REST);
    grid.active[4] = false;
    grid.cells[4].h = 0.0;
    grid
}

#[test]
fn port_faces_reject_with_named_face() -> TestResult {
    let solver = engine(holed())?;
    let right = PortReachEnd::Right;
    for (specs, fragment) in [
        (vec![], "no ports"),
        (vec![spec(vec![], 0.0, right)], "port=0 has no faces"),
        (
            vec![spec(vec![x_face(0, 0)], f64::NAN, right)],
            "supply a finite reach end bed",
        ),
        (
            vec![spec(vec![x_face(1, 0)], 0.0, right)],
            "face=BoundaryFace { axis: X, column: 1, row: 0 } has two active cells",
        ),
        (
            vec![spec(vec![y_face(1, 2)], 0.0, right)],
            "has no active surface cell",
        ),
        (
            vec![spec(vec![x_face(0, 0), y_face(0, 0)], 0.0, right)],
            "bind faces of one axis on one face line",
        ),
        (
            vec![spec(vec![x_face(0, 0), x_face(3, 1)], 0.0, right)],
            "bind faces of one axis on one face line",
        ),
        (
            vec![spec(vec![x_face(0, 0), x_face(0, 0)], 0.0, right)],
            "repeats: bind each face to one port once",
        ),
        (
            vec![
                spec(vec![x_face(0, 0)], 0.0, right),
                spec(vec![x_face(0, 0)], 0.0, PortReachEnd::Left),
            ],
            "port=1, face=BoundaryFace { axis: X, column: 0, row: 0 } repeats",
        ),
    ] {
        let message = rejection(solver.coupling_ports(&specs));
        assert!(message.contains(fragment), "{message} lacks {fragment}");
        assert!(message.contains("surface ports"), "{message}");
    }
    Ok(())
}

#[test]
fn caller_open_face_cannot_be_a_port_face() -> TestResult {
    let face = x_face(0, 0);
    let solver = engine_with(
        Grid::uniform(2, 1, [1.0, 1.0], 0.0, REST),
        vec![face],
        vec![
            FluxPort {
                face,
                water_rate: 0.0,
                momentum_rate: [0.0, 0.0],
                signal_speed: 10.0,
                provenance: "closed test port".into(),
            }
            .into(),
        ],
    )?;
    let message = rejection(solver.coupling_ports(&[spec(vec![face], 0.0, PortReachEnd::Right)]));
    assert!(message.contains("is a caller open face"), "{message}");
    // The other edge stays admissible.
    solver.coupling_ports(&[spec(vec![x_face(2, 0)], 0.0, PortReachEnd::Left)])?;
    Ok(())
}

#[test]
fn face_line_rejects_a_gap_and_a_side_change() -> TestResult {
    let column = engine(Grid::uniform(1, 3, [1.0, 1.0], 0.0, REST))?;
    let message = rejection(column.coupling_ports(&[spec(
        vec![x_face(0, 2), x_face(0, 0)],
        0.0,
        PortReachEnd::Right,
    )]));
    assert!(message.contains("with a gap"), "{message}");
    column.coupling_ports(&[spec(
        vec![x_face(0, 2), x_face(0, 0), x_face(0, 1)],
        0.0,
        PortReachEnd::Right,
    )])?;

    // Cells 0 and 3 are active. The row-1 face line meets cell 0 below and cell 3 above.
    let mut grid = Grid::uniform(2, 2, [1.0, 1.0], 0.0, REST);
    grid.active = vec![true, false, false, true];
    grid.cells[1].h = 0.0;
    grid.cells[2].h = 0.0;
    let diagonal = engine(grid)?;
    let message = rejection(diagonal.coupling_ports(&[spec(
        vec![y_face(0, 1), y_face(1, 1)],
        0.0,
        PortReachEnd::Right,
    )]));
    assert!(
        message.contains("bind faces with the surface on one side"),
        "{message}"
    );
    Ok(())
}

#[test]
fn ports_and_ghosts_reject_another_engine() -> TestResult {
    let solver = engine(holed())?;
    let other = engine(Grid::uniform(2, 1, [1.0, 1.0], 0.0, REST))?;
    let ports = other.coupling_ports(&[spec(vec![x_face(0, 0)], 0.0, PortReachEnd::Right)])?;
    let message = rejection(solver.coupling_port_ghosts(
        &ports,
        &numr::tensor::Tensor::from_slice(&[0.5], &[1], &numr::runtime::cpu::CpuDevice::new())?,
        &numr::tensor::Tensor::from_slice(&[0.0], &[1], &numr::runtime::cpu::CpuDevice::new())?,
    ));
    assert!(
        message.contains("build the ports from this engine"),
        "{message}"
    );
    let foreign = ghosts(&other, &ports, &[0.5], &[0.0])?;
    let message = rejection(prepare(&solver, 0.01, Some((&ports, &foreign))));
    assert!(
        message.contains("build the ports from this engine"),
        "{message}"
    );
    let own = solver.coupling_ports(&[spec(vec![x_face(0, 0)], 0.0, PortReachEnd::Right)])?;
    let message = rejection(prepare(&solver, 0.01, Some((&own, &foreign))));
    assert!(message.contains("another port set"), "{message}");
    Ok(())
}
