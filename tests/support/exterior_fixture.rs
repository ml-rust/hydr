#[path = "impervious.rs"]
mod impervious;

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    Axis, BoundaryConditions, BoundaryFace, BoundaryForcing, ExteriorBoundary, ExteriorState,
    ForcingInterval, ForcingSchedule, Solver, SolverConfig,
};
use impervious::impervious_solver;

pub type TestResult = Result<(), Box<dyn std::error::Error>>;

pub fn interval(
    start: f64,
    end: f64,
    count: usize,
    boundaries: Vec<BoundaryForcing>,
) -> ForcingInterval {
    ForcingInterval {
        start,
        end,
        potential_evaporation: vec![0.0; count],
        rain: vec![0.0; count],
        withdrawal: vec![0.0; count],
        boundaries,
    }
}

pub fn wet(
    face: BoundaryFace,
    elevation: f64,
    outward_velocity: f64,
    tangential_velocity: f64,
) -> BoundaryForcing {
    BoundaryForcing::Exterior(ExteriorBoundary {
        face,
        state: ExteriorState::Wet {
            elevation,
            outward_velocity,
            tangential_velocity,
        },
    })
}

pub fn create(
    nx: usize,
    ny: usize,
    bed: Vec<f64>,
    cells: Vec<CellState>,
    intervals: Vec<ForcingInterval>,
    config: SolverConfig,
) -> Result<Solver, Box<dyn std::error::Error>> {
    let domain = Domain::new(
        CartesianGrid::new(nx, ny, 1.0, 1.0)?,
        bed,
        vec![0.0; nx * ny],
        vec![true; nx * ny],
        SpatialReference::new(
            "synthetic projected metres".into(),
            "synthetic SI datum".into(),
            [0.0; 2],
        )?,
    )?;
    let faces = intervals
        .first()
        .ok_or("missing forcing interval")?
        .boundaries
        .iter()
        .map(BoundaryForcing::face)
        .collect();
    let boundaries = BoundaryConditions::with_open_faces(&domain, faces)?;
    let forcing = ForcingSchedule::new(&domain, &boundaries, intervals)?;
    let state = FlowState::new(&domain, cells)?;
    impervious_solver(domain, state, boundaries, forcing, config, 0.0)
}

pub fn uniform(
    nx: usize,
    ny: usize,
    bed: Vec<f64>,
    eta: f64,
    velocity: [f64; 2],
) -> Result<Solver, Box<dyn std::error::Error>> {
    let mut boundaries = Vec::new();
    for row in 0..ny {
        for column in [0, nx] {
            boundaries.push(wet(
                BoundaryFace {
                    axis: Axis::X,
                    column,
                    row,
                },
                eta,
                if column == 0 {
                    -velocity[0]
                } else {
                    velocity[0]
                },
                velocity[1],
            ));
        }
    }
    for column in 0..nx {
        for row in [0, ny] {
            boundaries.push(wet(
                BoundaryFace {
                    axis: Axis::Y,
                    column,
                    row,
                },
                eta,
                if row == 0 { -velocity[1] } else { velocity[1] },
                velocity[0],
            ));
        }
    }
    let cells = bed
        .iter()
        .map(|bed| {
            let h = eta - bed;
            CellState {
                h,
                qx: h * velocity[0],
                qy: h * velocity[1],
            }
        })
        .collect();
    create(
        nx,
        ny,
        bed,
        cells,
        vec![interval(0.0, 1.0, nx * ny, boundaries)],
        SolverConfig {
            max_step: 0.01,
            ..SolverConfig::default()
        },
    )
}

pub fn through(axis: Axis, direction: f64) -> Result<Solver, Box<dyn std::error::Error>> {
    let (upstream, downstream, cell) = match axis {
        Axis::X => (
            BoundaryFace {
                axis,
                column: if direction > 0.0 { 0 } else { 1 },
                row: 0,
            },
            BoundaryFace {
                axis,
                column: if direction > 0.0 { 1 } else { 0 },
                row: 0,
            },
            CellState {
                h: 1.0,
                qx: 5.0 * direction,
                qy: 0.0,
            },
        ),
        Axis::Y => (
            BoundaryFace {
                axis,
                column: 0,
                row: if direction > 0.0 { 0 } else { 1 },
            },
            BoundaryFace {
                axis,
                column: 0,
                row: if direction > 0.0 { 1 } else { 0 },
            },
            CellState {
                h: 1.0,
                qx: 0.0,
                qy: 5.0 * direction,
            },
        ),
    };
    create(
        1,
        1,
        vec![0.0],
        vec![cell],
        vec![interval(
            0.0,
            1.0,
            1,
            vec![
                wet(upstream, 1.0, -5.0, 0.0),
                BoundaryForcing::Transmissive(downstream),
            ],
        )],
        SolverConfig {
            max_step: 0.01,
            ..SolverConfig::default()
        },
    )
}
