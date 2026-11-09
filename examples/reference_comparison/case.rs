use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    AdvanceStatus, BoundaryConditions, ForcingInterval, ForcingSchedule, Solver, SolverConfig,
};

pub(super) type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(super) const LENGTH: f64 = 200.0;
pub(super) const END: f64 = 4.0;
pub(super) const STEP: f64 = 0.01;

#[derive(Clone, Copy)]
pub(super) struct Case {
    pub name: &'static str,
    pub downstream: f64,
}

impl Case {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "wet-quarter" => Ok(Self {
                name: "wet-quarter",
                downstream: 0.25,
            }),
            "wet-tenth" => Ok(Self {
                name: "wet-tenth",
                downstream: 0.1,
            }),
            _ => Err(format!("case={value}: choose wet-quarter or wet-tenth").into()),
        }
    }

    pub fn initial_storage(self) -> f64 {
        0.5 * LENGTH * (1.0 + self.downstream)
    }

    pub fn front_threshold(self) -> f64 {
        self.downstream + 0.05 * (1.0 - self.downstream)
    }
}

pub(super) fn centre(index: usize, cells: usize) -> f64 {
    -0.5 * LENGTH + (index as f64 + 0.5) * LENGTH / cells as f64
}

pub(super) fn simulate(case: Case, cells: usize) -> Result<Solver> {
    let domain = Domain::new(
        CartesianGrid::new(cells, 1, LENGTH / cells as f64, 1.0)?,
        vec![0.0; cells],
        vec![0.0; cells],
        vec![true; cells],
        SpatialReference::new(
            "synthetic projected metres".into(),
            "synthetic SI datum".into(),
            [-100.0, 0.0],
        )?,
    )?;
    let initial = (0..cells)
        .map(|index| CellState {
            h: if centre(index, cells) < 0.0 {
                1.0
            } else {
                case.downstream
            },
            qx: 0.0,
            qy: 0.0,
        })
        .collect();
    let state = FlowState::new(&domain, initial)?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: END + 1.0,
            rain: vec![0.0; cells],
            withdrawal: vec![0.0; cells],
            ports: vec![],
        }],
    )?;
    let mut solver = Solver::new(
        domain,
        state,
        boundaries,
        forcing,
        SolverConfig {
            max_step: STEP,
            ..SolverConfig::default()
        },
        0.0,
    )?;
    let mut dt_min = f64::INFINITY;
    let mut dt_max = 0.0_f64;
    let mut strictly_wet = true;
    let report = solver.advance_controlled(
        END,
        100_000,
        || false,
        |step, cells| {
            dt_min = dt_min.min(step.dt);
            dt_max = dt_max.max(step.dt);
            strictly_wet &= cells.iter().all(|cell| cell.h > 0.0);
        },
    )?;
    if report.status != AdvanceStatus::ReachedTarget {
        return Err(format!(
            "hydr endpoint={}: require complete advancement to {END}",
            report.accepted_time
        )
        .into());
    }
    let ledger = solver.ledger();
    let tolerance = 1e-9 + 1e-12 * ledger.initial_storage.max(ledger.storage).max(1.0);
    if ledger.residual.abs() > tolerance || ledger.water_adjustment != 0.0 {
        return Err(
            format!("hydr ledger={ledger:?}: require conserved water without adjustment").into(),
        );
    }
    if !strictly_wet {
        return Err(format!(
            "hydr case={}: require strictly positive accepted wet-bed depth",
            case.name
        )
        .into());
    }
    println!(
        "hydr_accepted_steps={} hydr_time={:.17e} hydr_dt_min={dt_min:.17e} hydr_dt_max={dt_max:.17e}",
        report.accepted_steps, report.accepted_time
    );
    Ok(solver)
}
