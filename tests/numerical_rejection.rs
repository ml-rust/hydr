//! Checks numerical rejections retain accepted state and actionable face context.

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    Axis, BoundaryConditions, BoundaryFace, FluxPort, ForcingInterval, ForcingSchedule, Solver,
    SolverConfig,
};

fn domain() -> Result<Domain, Box<dyn std::error::Error>> {
    Ok(Domain::new(
        CartesianGrid::new(2, 1, 1.0, 1.0)?,
        vec![0.0; 2],
        vec![0.0; 2],
        vec![true; 2],
        SpatialReference::new(
            "synthetic metric plane".into(),
            "synthetic datum".into(),
            [0.0; 2],
        )?,
    )?)
}

fn forcing(
    domain: &Domain,
    walls: &BoundaryConditions,
    ports: Vec<FluxPort>,
) -> Result<ForcingSchedule, Box<dyn std::error::Error>> {
    Ok(ForcingSchedule::new(
        domain,
        walls,
        vec![ForcingInterval {
            start: 0.0,
            end: 1.0,
            rain: vec![0.0; 2],
            withdrawal: vec![0.0; 2],
            boundaries: ports.into_iter().map(Into::into).collect(),
        }],
    )?)
}

#[test]
fn exhausted_face_bracket_identifies_support_and_preserves_state()
-> Result<(), Box<dyn std::error::Error>> {
    let domain = domain()?;
    let state = FlowState::new(
        &domain,
        vec![
            CellState {
                h: 1.0,
                ..CellState::default()
            },
            CellState {
                h: 0.25,
                ..CellState::default()
            },
        ],
    )?;
    let walls = BoundaryConditions::impermeable_walls();
    let input = forcing(&domain, &walls, vec![])?;
    let mut solver = Solver::new(
        domain,
        state,
        walls,
        input,
        SolverConfig {
            root_iterations: 1,
            ..SolverConfig::default()
        },
        0.0,
    )?;
    let before = solver.snapshot();
    let error = solver
        .advance_to(0.1, 10)
        .err()
        .ok_or("expected root exhaustion")?;
    let message = error.to_string();
    assert!(message.contains("column: 1"), "{message}");
    assert!(message.contains("iteration limit"), "{message}");
    assert!(message.contains("[h,qn,qt,bed]"), "{message}");
    assert_eq!(solver.snapshot(), before);
    Ok(())
}

#[test]
fn port_bounds_must_include_known_interior_characteristics()
-> Result<(), Box<dyn std::error::Error>> {
    let domain = domain()?;
    let state = FlowState::new(
        &domain,
        vec![
            CellState {
                h: 1.0,
                ..CellState::default()
            };
            2
        ],
    )?;
    let face = BoundaryFace {
        axis: Axis::X,
        column: 0,
        row: 0,
    };
    let ports = BoundaryConditions::with_open_faces(&domain, vec![face])?;
    let input = forcing(
        &domain,
        &ports,
        vec![FluxPort {
            face,
            water_rate: 0.0,
            momentum_rate: [-4.905, 0.0],
            signal_speed: 0.1,
            provenance: "deliberately insufficient manufactured bound".into(),
        }],
    )?;
    let mut solver = Solver::new(domain, state, ports, input, SolverConfig::default(), 0.0)?;
    let before = solver.snapshot();
    let error = solver
        .advance_to(0.01, 10)
        .err()
        .ok_or("expected inadmissible port bound")?;
    assert!(error.to_string().contains("interior characteristic"));
    assert_eq!(solver.snapshot(), before);
    Ok(())
}
