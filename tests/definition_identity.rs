//! Checks bit-sensitive immutable identities before checkpoint restoration.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "support/impervious.rs"]
mod impervious;

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{BoundaryConditions, ForcingInterval, ForcingSchedule, Solver, SolverConfig};
use impervious::impervious_solver;

fn create(
    film: f64,
    rain: f64,
) -> Result<Solver<crate::runtime::Runtime>, Box<dyn std::error::Error>> {
    let domain = Domain::new(
        CartesianGrid::new(1, 1, 1.0, 1.0)?,
        vec![0.0],
        vec![0.0],
        vec![true],
        SpatialReference::new(
            "synthetic metric plane".into(),
            "synthetic datum".into(),
            [0.0; 2],
        )?,
    )?;
    let state = FlowState::new(&domain, vec![CellState::default()])?;
    let walls = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &walls,
        vec![ForcingInterval {
            start: 0.0,
            end: 1.0,
            potential_evaporation: vec![0.0],
            rain: vec![rain],
            withdrawal: vec![0.0],
            boundaries: vec![],
        }],
    )?;
    impervious_solver(
        domain,
        state,
        walls,
        forcing,
        SolverConfig {
            momentum_film_depth: film,
            ..SolverConfig::default()
        },
        0.0,
    )
}

#[test]
fn signed_zero_definition_changes_reject_both_restore_paths()
-> Result<(), Box<dyn std::error::Error>> {
    let source = create(0.0, 0.0)?;
    let snapshot = source.snapshot();
    for (film, rain) in [(-0.0, 0.0), (0.0, -0.0)] {
        let mut destination = create(film, rain)?;
        let before = destination.snapshot();
        assert!(destination.restore(&snapshot).is_err());
        assert_eq!(destination.snapshot().data()?, before.data()?);
        assert!(destination.restore_data(&snapshot.data()?).is_err());
        assert_eq!(destination.snapshot().data()?, before.data()?);
    }
    Ok(())
}
