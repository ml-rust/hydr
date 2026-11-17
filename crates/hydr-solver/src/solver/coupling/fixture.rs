//! Test engines and exchange operands for the coordinator API.

use super::{CouplingCompletion, CouplingPreparation, ExchangeSource, SurfaceLinks};
use crate::contract::{BoundaryConditions, ForcingInterval, ForcingSchedule, SolverConfig};
use crate::grid::CartesianGrid;
use crate::model::{CellState, Domain, FlowState, SpatialReference};
use crate::soil::{LandModel, SoilState};
use crate::solver::{InitialState, Solver, SolverSetup};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

pub(in crate::solver) type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Cell area of [`lake`] in m².
pub(in crate::solver) const AREA: f64 = 4.0;

/// Three frictionless 2 m cells of a flat walled lake, 0.5 m deep, with uniform x discharge.
pub(in crate::solver) fn lake() -> TestResult<Solver<CpuRuntime>> {
    let domain = Domain::new(
        CartesianGrid::new(3, 1, 2.0, 2.0)?,
        vec![0.0; 3],
        vec![0.0; 3],
        vec![true; 3],
        SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
    )?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 10.0,
            rain: vec![0.0; 3],
            potential_evaporation: vec![0.0; 3],
            withdrawal: vec![0.0; 3],
            boundaries: Vec::new(),
        }],
    )?;
    let land = LandModel::impervious(&domain);
    let soil = SoilState::initialized(&land)?;
    let flow = FlowState::new(
        &domain,
        vec![
            CellState {
                h: 0.5,
                qx: 0.05,
                qy: 0.01,
            };
            3
        ],
    )?;
    Ok(Solver::<CpuRuntime>::new(
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None)),
        SolverSetup {
            domain,
            land,
            boundaries,
            forcing,
            config: SolverConfig::default(),
        },
        InitialState {
            time: 0.0,
            flow,
            soil,
        },
    )?)
}

/// Builds exchange operands over `links` from host lanes.
pub(in crate::solver) fn exchange(
    links: &SurfaceLinks<CpuRuntime>,
    inflow: &[f64],
    withdrawal: &[f64],
    velocity: [&[f64]; 2],
) -> TestResult<ExchangeSource<CpuRuntime>> {
    let device = CpuDevice::new();
    let lane = |values: &[f64]| Tensor::<CpuRuntime>::from_slice(values, &[values.len()], &device);
    Ok(ExchangeSource::new(
        links,
        lane(inflow)?,
        lane(withdrawal)?,
        [lane(velocity[0])?, lane(velocity[1])?],
    )?)
}

/// Returns the accepted engine of one coupled step to `end`, or the error text.
pub(in crate::solver) fn coupled_step(
    solver: &Solver<CpuRuntime>,
    end: f64,
    exchange: Option<&ExchangeSource<CpuRuntime>>,
    links: &SurfaceLinks<CpuRuntime>,
) -> TestResult<Solver<CpuRuntime>> {
    let CouplingPreparation::Ready(stage) = solver.coupling_prepare(end, 0, links)? else {
        return Err("the lake step must prepare without retry".into());
    };
    match solver.coupling_complete(&stage, exchange)? {
        CouplingCompletion::Accepted(next) => Ok(*next),
        CouplingCompletion::Retry(reason) => Err(reason.into()),
    }
}
