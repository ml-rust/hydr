//! Advances a synthetic impermeable basin with explicitly supplied rainfall.

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{BoundaryConditions, ForcingInterval, ForcingSchedule, Solver, SolverConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid = CartesianGrid::new(8, 8, 2.0, 2.0)?;
    let count = grid.cell_count();
    let reference = SpatialReference::new(
        "synthetic metric plane".into(),
        "synthetic datum".into(),
        [0.0, 0.0],
    )?;
    let domain = Domain::new(
        grid,
        vec![0.0; count],
        vec![0.03; count],
        vec![true; count],
        reference,
    )?;
    let state = FlowState::new(&domain, vec![CellState::default(); count])?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 60.0,
            rain: vec![1e-5; count],
            withdrawal: vec![0.0; count],
            boundaries: vec![],
        }],
    )?;
    let mut solver = Solver::new(
        domain,
        state,
        boundaries,
        forcing,
        SolverConfig::default(),
        0.0,
    )?;
    let report = solver.advance_to(60.0, 10_000)?;
    println!(
        "Status: {:?}, accepted time: {} s",
        report.status, report.accepted_time
    );
    println!("Storage: {} m³", solver.ledger().storage);
    println!("Water residual: {} m³", solver.ledger().residual);
    Ok(())
}
