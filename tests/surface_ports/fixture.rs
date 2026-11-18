//! Walled frictionless grids, port declarations, and coordinator stage steps.

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    Axis, BoundaryConditions, BoundaryFace, BoundaryForcing, CellResult, ForcingInterval,
    ForcingSchedule, InitialState, Solver, SolverConfig, SolverSetup,
};
use hydr_solver::internal::{
    CouplingCompletion, CouplingPreparation, PortGhosts, PortReachEnd, SurfaceCouplingStage,
    SurfacePortSpec, SurfacePorts,
};
use numr::tensor::Tensor;

pub type Runtime = crate::runtime::Runtime;
pub type Engine = Solver<Runtime>;
pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// Gravity of [`SolverConfig::default`], in m/s².
pub const GRAVITY: f64 = 9.81;

/// Grid layout and initial cell states, in row-major order.
pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    pub dx: f64,
    pub dy: f64,
    pub bed: Vec<f64>,
    pub active: Vec<bool>,
    pub cells: Vec<CellState>,
}

impl Grid {
    /// An all-active grid on one bed with one cell state everywhere.
    pub fn uniform(nx: usize, ny: usize, spacing: [f64; 2], bed: f64, cell: CellState) -> Self {
        Self {
            nx,
            ny,
            dx: spacing[0],
            dy: spacing[1],
            bed: vec![bed; nx * ny],
            active: vec![true; nx * ny],
            cells: vec![cell; nx * ny],
        }
    }
}

/// A frictionless impervious engine over `grid` with walls and the given open-face forcing.
pub fn engine_with(
    grid: Grid,
    open: Vec<BoundaryFace>,
    forcing: Vec<BoundaryForcing>,
) -> TestResult<Engine> {
    let count = grid.nx * grid.ny;
    let domain = Domain::new(
        CartesianGrid::new(grid.nx, grid.ny, grid.dx, grid.dy)?,
        grid.bed,
        vec![0.0; count],
        grid.active,
        SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
    )?;
    let boundaries = BoundaryConditions::with_open_faces(&domain, open)?;
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 1.0e3,
            rain: vec![0.0; count],
            potential_evaporation: vec![0.0; count],
            withdrawal: vec![0.0; count],
            boundaries: forcing,
        }],
    )?;
    let land = LandModel::impervious(&domain);
    let soil = SoilState::initialized(&land)?;
    let flow = FlowState::new(&domain, grid.cells)?;
    Ok(Solver::<Runtime>::new(
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
            flow,
            soil,
        },
    )?)
}

/// A walled engine over `grid`.
pub fn engine(grid: Grid) -> TestResult<Engine> {
    engine_with(grid, Vec::new(), Vec::new())
}

pub fn x_face(column: usize, row: usize) -> BoundaryFace {
    BoundaryFace {
        axis: Axis::X,
        column,
        row,
    }
}

pub fn y_face(column: usize, row: usize) -> BoundaryFace {
    BoundaryFace {
        axis: Axis::Y,
        column,
        row,
    }
}

pub fn spec(faces: Vec<BoundaryFace>, bed: f64, end: PortReachEnd) -> SurfacePortSpec {
    SurfacePortSpec {
        faces,
        ghost_bed_m: bed,
        reach_end: end,
    }
}

/// Builds ghosts from per-port host depth and reach-positive discharge per width.
pub fn ghosts(
    solver: &Engine,
    ports: &SurfacePorts<Runtime>,
    depth: &[f64],
    discharge: &[f64],
) -> TestResult<PortGhosts<Runtime>> {
    let device = numr::runtime::cpu::CpuDevice::new();
    let lane = |values: &[f64]| Tensor::<Runtime>::from_slice(values, &[values.len()], &device);
    Ok(solver.coupling_port_ghosts(ports, &lane(depth)?, &lane(discharge)?)?)
}

/// Prepares a stage of at most `dt`. A step past the stage limit retries at `0.9` of it.
pub fn prepare(
    solver: &Engine,
    dt: f64,
    ports: Option<(&SurfacePorts<Runtime>, &PortGhosts<Runtime>)>,
) -> TestResult<SurfaceCouplingStage<Runtime>> {
    let links = solver.coupling_links(&[first_active(solver)?])?;
    let mut step = dt;
    for _ in 0..2 {
        let end = solver.time() + step;
        let CouplingPreparation::Ready(stage) = solver.coupling_prepare(end, 0, &links, ports)?
        else {
            return Err("the stage must prepare without a retry".into());
        };
        if stage.dt() <= stage.limit() {
            return Ok(*stage);
        }
        step = 0.9 * stage.limit();
    }
    Err("the stage exceeds its own limit after one retry".into())
}

/// Completes `stage` without a link exchange.
pub fn complete(solver: &Engine, stage: &SurfaceCouplingStage<Runtime>) -> TestResult<Engine> {
    match solver.coupling_complete(stage, None)? {
        CouplingCompletion::Accepted(next) => Ok(*next),
        CouplingCompletion::Retry(reason) => Err(reason.into()),
    }
}

fn first_active(solver: &Engine) -> TestResult<usize> {
    let state = solver.state()?;
    (0..state.cells().len())
        .find(|&cell| {
            solver
                .cell(cell)
                .is_ok_and(|result| result != CellResult::Unsupported)
        })
        .ok_or_else(|| "the grid has no active cell".into())
}

/// Copies an F64 tensor to the host.
pub fn host(tensor: &Tensor<Runtime>) -> TestResult<Vec<f64>> {
    Ok(tensor.try_to_vec::<f64>()?)
}

/// Relative gap `|a - b| / max(|a|, |b|)`, zero when both are zero.
pub fn relative(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs());
    if scale == 0.0 {
        0.0
    } else {
        (a - b).abs() / scale
    }
}

/// Error text of a rejected result.
pub fn rejection<T>(result: Result<T, impl std::fmt::Display>) -> String {
    match result {
        Ok(_) => String::new(),
        Err(error) => error.to_string(),
    }
}
