use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    AdvanceStatus, BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, Solver,
    SolverConfig, SolverSetup,
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
            "dry-ritter" => Ok(Self {
                name: "dry-ritter",
                downstream: 0.0,
            }),
            _ => Err(format!("case={value}: choose wet-quarter, wet-tenth, or dry-ritter").into()),
        }
    }

    pub fn initial_storage(self) -> f64 {
        0.5 * LENGTH * (1.0 + self.downstream)
    }

    pub fn front_threshold(self) -> f64 {
        if self.allows_dry() {
            1e-3
        } else {
            self.downstream + 0.05 * (1.0 - self.downstream)
        }
    }

    pub fn allows_dry(self) -> bool {
        self.downstream == 0.0
    }

    pub fn reference_method(self) -> &'static str {
        if self.allows_dry() {
            "pyclaw-fortran-augmented-order1"
        } else {
            "pyclaw-python-hll-order1"
        }
    }
}

pub(super) fn centre(index: usize, cells: usize) -> f64 {
    -0.5 * LENGTH + (index as f64 + 0.5) * LENGTH / cells as f64
}

pub(super) fn simulate(case: Case, cells: usize) -> Result<Solver<crate::runtime::Runtime>> {
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
            potential_evaporation: vec![0.0; cells],
            rain: vec![0.0; cells],
            withdrawal: vec![0.0; cells],
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
            config: SolverConfig {
                max_step: STEP,
                ..SolverConfig::default()
            },
        },
        InitialState {
            time: 0.0,
            flow: state,
            soil,
        },
    )?;
    let mut dt_min = f64::INFINITY;
    let mut dt_max = 0.0_f64;
    let mut admitted_states = true;
    let mut max_abs_qy = 0.0_f64;
    let report = solver.advance_controlled(
        END,
        100_000,
        || false,
        |step, cells| {
            dt_min = dt_min.min(step.dt);
            dt_max = dt_max.max(step.dt);
            admitted_states &= cells.iter().all(|cell| {
                cell.h >= 0.0
                    && (case.allows_dry() || cell.h > 0.0)
                    && (cell.h != 0.0 || (cell.qx == 0.0 && cell.qy == 0.0))
                    && cell.qy.abs() <= 1e-12
            });
            for cell in cells {
                max_abs_qy = max_abs_qy.max(cell.qy.abs());
            }
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
    if !admitted_states {
        return Err(format!(
            "hydr case={}: require admitted wet/dry state, zero dry momentum, and |qy|<=1e-12",
            case.name
        )
        .into());
    }
    println!(
        "hydr_accepted_steps={} hydr_time={:.17e} hydr_dt_min={dt_min:.17e} hydr_dt_max={dt_max:.17e} hydr_max_abs_qy={max_abs_qy:.17e}",
        report.accepted_steps, report.accepted_time
    );
    Ok(solver)
}
