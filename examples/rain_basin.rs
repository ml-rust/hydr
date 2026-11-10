//! Advances a synthetic impermeable basin with explicitly supplied rainfall.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, Solver, SolverConfig,
    SolverSetup,
};

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
            potential_evaporation: vec![0.0; count],
            rain: vec![1e-5; count],
            withdrawal: vec![0.0; count],
            boundaries: vec![],
        }],
    )?;
    let land = LandModel::impervious(&domain);
    let soil = SoilState::initialized(&land)?;
    let mut solver = Solver::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        SolverSetup {
            domain,
            land,
            boundaries,
            forcing,
            config: SolverConfig::default(),
        },
        InitialState {
            time: 0.0,
            flow: state,
            soil,
        },
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
