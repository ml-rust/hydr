use hydr::model::{Domain, FlowState};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    BoundaryConditions, ForcingSchedule, InitialState, Solver, SolverConfig, SolverSetup,
};

pub(super) fn impervious_solver(
    domain: Domain,
    flow: FlowState,
    boundaries: BoundaryConditions,
    forcing: ForcingSchedule,
    config: SolverConfig,
    time: f64,
) -> Result<Solver<crate::runtime::Runtime>, Box<dyn std::error::Error>> {
    let land = LandModel::impervious(&domain);
    let soil = SoilState::initialized(&land)?;
    Ok(Solver::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        SolverSetup {
            domain,
            land,
            boundaries,
            forcing,
            config,
        },
        InitialState { time, flow, soil },
    )?)
}
