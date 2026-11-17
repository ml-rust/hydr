//! Native worker lifecycle: capability check, cancellation, checkpoint, restore, and forecast cycle.
//!
//! The `native_worker` example prints this lifecycle. The `native_worker` test asserts it.

use crate::impervious::impervious_solver;
use hydr::capability::{
    BoundaryKind, EndpointId, Precision, RuntimeKind, RuntimeStatus, SourceKind,
};
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    AdvanceReport, AdvanceStatus, Axis, BoundaryConditions, BoundaryFace, CONTRACT_VERSION,
    FluxPort, ForcingInterval, ForcingSchedule, SNAPSHOT_SCHEMA, SnapshotData, Solver,
    SolverConfig, StepDiagnostics, WaterLedger, surface_solver_capabilities,
};
use std::sync::atomic::{AtomicBool, Ordering};

/// Boxed error returned by every lifecycle step.
pub type WorkerError = Box<dyn std::error::Error>;

/// Engine on the sequential CPU reference runtime.
pub type Engine = Solver<crate::runtime::Runtime>;

/// Interval edges in seconds. Cycle 1 covers `[0, 2)`. Cycle 2 appends `[2, 3)`.
pub const EDGES: [f64; 4] = [0.0, 1.0, 2.0, 3.0];

/// Accepted steps after which the observer requests cancellation.
pub const CANCEL_AFTER: usize = 30;

/// Accepted-step budget for every advancement call.
pub const MAX_STEPS: usize = 10_000;

/// Boundary kinds this worker declares.
pub const BOUNDARIES: [BoundaryKind; 2] = [BoundaryKind::Wall, BoundaryKind::FluxPort];

/// Local sources this worker supplies.
pub const SOURCES: [SourceKind; 2] = [SourceKind::Rain, SourceKind::ImperviousLand];

/// Grid side length in cells.
const SIDE: usize = 4;

/// The single open face: the west face of cell (0, 0).
const PORT: BoundaryFace = BoundaryFace {
    axis: Axis::X,
    column: 0,
    row: 0,
};

/// Outcome of the admission check against `surface_solver_capabilities()`.
#[derive(Clone, Debug)]
pub struct CapabilityCheck {
    /// Stable ids of the admitted boundary kinds.
    pub boundaries: Vec<&'static str>,
    /// Stable ids of the admitted sources.
    pub sources: Vec<&'static str>,
    /// Stable id of the CPU runtime status.
    pub cpu_status: &'static str,
    /// Declared contract identity.
    pub contract: &'static str,
    /// Declared snapshot schema identity.
    pub schema: &'static str,
    /// Limit the provider accepts by declaring a flux port.
    pub port_limit: &'static str,
}

/// One accepted step recorded by the observer.
#[derive(Clone, Debug, PartialEq)]
pub struct AcceptedRecord {
    /// One-based accepted-step count across both controlled calls.
    pub step: usize,
    /// Accepted end time in seconds.
    pub time: f64,
    /// Water residual of this step, in m³.
    pub residual: f64,
    /// Largest accepted cell depth in metres.
    pub max_depth: f64,
}

/// Bitwise-comparable result of one engine at one accepted time.
#[derive(Clone, Debug, PartialEq)]
pub struct RunRecord {
    /// Owned checkpoint package.
    pub data: SnapshotData,
    /// Cumulative ledger.
    pub ledger: WaterLedger,
    /// Last accepted step.
    pub last_step: Option<StepDiagnostics>,
}

impl RunRecord {
    /// Captures package, ledger, and last step of `engine`.
    pub fn capture(engine: &Engine) -> Result<Self, WorkerError> {
        Ok(Self {
            data: engine.snapshot().data()?,
            ledger: engine.ledger().clone(),
            last_step: engine.last_step()?,
        })
    }

    /// Returns `true` when every field matches bit for bit.
    ///
    /// `Debug` prints each `f64` in its shortest round-trip form, so `-0.0` and `0.0` differ.
    pub fn same_bits(&self, other: &Self) -> bool {
        self == other && format!("{self:?}") == format!("{other:?}")
    }
}

/// Water budget of an accepted ledger, in m³.
#[derive(Clone, Debug, PartialEq)]
pub struct WaterBudget {
    /// Storage minus initial storage.
    pub storage_change: f64,
    /// Rain plus inward port water.
    pub inputs: f64,
    /// Outward port water, withdrawal, infiltration, and evaporation.
    pub outputs: f64,
    /// Signed numerical water adjustments.
    pub adjustment: f64,
    /// Signed cumulative residual.
    pub residual: f64,
    /// Documented residual bound: `volume_absolute + volume_relative * scale`, scale floor 1 m³.
    pub residual_bound: f64,
    /// Consumed absolute adjustment budget.
    pub adjustment_used: f64,
    /// Configured cumulative adjustment limit.
    pub adjustment_limit: f64,
}

impl WaterBudget {
    /// Builds the budget from `ledger` under `config`.
    pub fn new(ledger: &WaterLedger, config: &SolverConfig) -> Self {
        let scale = [
            ledger.storage,
            ledger.rain,
            ledger.withdrawal,
            ledger.infiltration,
            ledger.evaporation,
            ledger.water_adjustment.abs(),
            ledger.boundary_in,
            ledger.boundary_out,
        ]
        .into_iter()
        .fold(ledger.initial_storage.max(1.0), f64::max);
        Self {
            storage_change: ledger.storage - ledger.initial_storage,
            inputs: ledger.rain + ledger.boundary_in,
            outputs: ledger.boundary_out
                + ledger.withdrawal
                + ledger.infiltration
                + ledger.evaporation,
            adjustment: ledger.water_adjustment,
            residual: ledger.residual,
            residual_bound: config.volume_absolute + config.volume_relative * scale,
            adjustment_used: ledger.water_adjustment_abs_bound,
            adjustment_limit: config.max_cumulative_water_adjustment_m3,
        }
    }
}

/// Everything the lifecycle produces, in order.
#[derive(Clone, Debug)]
pub struct WorkerReport {
    /// Admission check outcome.
    pub capabilities: CapabilityCheck,
    /// Report of the cancelled controlled call.
    pub cancelled: AdvanceReport,
    /// Package persisted at cancellation.
    pub cancel_package: SnapshotData,
    /// Report of the controlled call that resumes to the cycle boundary.
    pub resumed: AdvanceReport,
    /// Observer records across both controlled calls.
    pub records: Vec<AcceptedRecord>,
    /// Uninterrupted cycle-1 run at the cycle boundary.
    pub uninterrupted: RunRecord,
    /// Cancelled-then-resumed worker at the cycle boundary.
    pub worker: RunRecord,
    /// Fresh engine after `restore_data` of the cancellation package, at the cycle boundary.
    pub restored: RunRecord,
    /// `Solver::resume` of the cancellation package, at the cycle boundary.
    pub resumed_package: RunRecord,
    /// Single run over the full spliced schedule, at the forecast end.
    pub single_forecast: RunRecord,
    /// Each continued engine at the forecast end, labelled.
    pub forecasts: Vec<(&'static str, RunRecord)>,
    /// Water budget of the worker at the forecast end.
    pub budget: WaterBudget,
}

/// Refuses the run unless the surface solver admits every declared boundary and source.
///
/// # Errors
/// Names the first refused requirement and the next action.
pub fn check_capabilities() -> Result<CapabilityCheck, WorkerError> {
    let caps = surface_solver_capabilities();
    for kind in BOUNDARIES {
        if !caps.admits(EndpointId::ExteriorFace, kind) {
            return Err(format!(
                "boundary {} refused at endpoint {}: remove it from the worker definition",
                kind.id(),
                EndpointId::ExteriorFace.id()
            )
            .into());
        }
    }
    for source in SOURCES {
        if !caps.sources.contains(&source) {
            return Err(
                format!("source {} refused: remove it from the forcing", source.id()).into(),
            );
        }
    }
    if !caps.precision.contains(&Precision::F64) {
        return Err("precision f64 refused: select an F64 build".into());
    }
    let cpu = caps
        .runtime(RuntimeKind::Cpu)
        .filter(|support| support.status == RuntimeStatus::Reference)
        .ok_or("runtime cpu is not the reference: select the CPU reference runtime")?;
    let contract = caps
        .contract
        .filter(|contract| *contract == CONTRACT_VERSION)
        .ok_or("contract mismatch: rebuild the worker against this hydr build")?;
    let schema = caps
        .snapshot_schema
        .filter(|schema| *schema == SNAPSHOT_SCHEMA)
        .ok_or("snapshot schema mismatch: rebuild the worker against this hydr build")?;
    let port_limit = caps
        .limit_by_id("flux-port-provider-attested")
        .ok_or("limit flux-port-provider-attested is undeclared: recheck flux-port admission")?;
    Ok(CapabilityCheck {
        boundaries: BOUNDARIES.iter().map(BoundaryKind::id).collect(),
        sources: SOURCES.iter().map(SourceKind::id).collect(),
        cpu_status: cpu.status.id(),
        contract,
        schema,
        port_limit: port_limit.summary,
    })
}

fn domain() -> Result<Domain, WorkerError> {
    Ok(Domain::new(
        CartesianGrid::new(SIDE, SIDE, 1.0, 1.0)?,
        vec![0.0; SIDE * SIDE],
        vec![0.03; SIDE * SIDE],
        vec![true; SIDE * SIDE],
        SpatialReference::new(
            "synthetic metric plane".into(),
            "synthetic datum".into(),
            [0.0; 2],
        )?,
    )?)
}

/// Returns interval `k` with rain and one manufactured source-complete inflow port.
///
/// The port carries the hydrostatic momentum flux of a 1 m resting depth over a 1 m face.
fn interval(k: usize) -> ForcingInterval {
    let scale = (k + 1) as f64;
    ForcingInterval {
        start: EDGES[k],
        end: EDGES[k + 1],
        rain: vec![1e-4 * scale; SIDE * SIDE],
        potential_evaporation: vec![0.0; SIDE * SIDE],
        withdrawal: vec![0.0; SIDE * SIDE],
        boundaries: vec![
            FluxPort {
                face: PORT,
                water_rate: -1e-3 * scale,
                momentum_rate: [-4.905, 0.0],
                signal_speed: 10.0,
                provenance: "manufactured full flux, h near 1 m, speed bound 10 m/s".into(),
            }
            .into(),
        ],
    }
}

/// Returns the checked schedule of intervals `range`.
pub fn schedule(range: std::ops::Range<usize>) -> Result<ForcingSchedule, WorkerError> {
    let domain = domain()?;
    let boundaries = BoundaryConditions::with_open_faces(&domain, vec![PORT])?;
    Ok(ForcingSchedule::new(
        &domain,
        &boundaries,
        range.map(interval).collect(),
    )?)
}

/// Builds a resting 1 m basin at time zero under intervals `range`.
pub fn engine(range: std::ops::Range<usize>) -> Result<Engine, WorkerError> {
    let domain = domain()?;
    let boundaries = BoundaryConditions::with_open_faces(&domain, vec![PORT])?;
    let state = FlowState::new(
        &domain,
        vec![
            CellState {
                h: 1.0,
                ..CellState::default()
            };
            SIDE * SIDE
        ],
    )?;
    let forcing = ForcingSchedule::new(&domain, &boundaries, range.map(interval).collect())?;
    impervious_solver(
        domain,
        state,
        boundaries,
        forcing,
        SolverConfig::default(),
        0.0,
    )
}

/// Advances `engine` to `target` and requires completion.
fn complete(engine: &mut Engine, target: f64) -> Result<(), WorkerError> {
    let report = engine.advance_to(target, MAX_STEPS)?;
    if report.status != AdvanceStatus::ReachedTarget {
        return Err(format!(
            "advance to {target} s stopped as {:?} at {} s: raise MAX_STEPS",
            report.status, report.accepted_time
        )
        .into());
    }
    Ok(())
}

/// Runs one controlled call. The observer records each step and cancels after `CANCEL_AFTER`.
fn controlled(
    engine: &mut Engine,
    cancel: &AtomicBool,
    records: &mut Vec<AcceptedRecord>,
) -> Result<AdvanceReport, WorkerError> {
    Ok(engine.advance_controlled(
        EDGES[2],
        MAX_STEPS,
        || cancel.load(Ordering::SeqCst),
        |step, cells| {
            records.push(AcceptedRecord {
                step: records.len() + 1,
                time: step.end,
                residual: step.residual,
                max_depth: cells.iter().map(|cell| cell.h).fold(0.0, f64::max),
            });
            if records.len() == CANCEL_AFTER {
                cancel.store(true, Ordering::SeqCst);
            }
        },
    )?)
}

/// Runs the complete worker lifecycle on the CPU reference runtime.
///
/// # Errors
/// Returns capability refusals, construction, advancement, checkpoint, and continuation errors.
pub fn run() -> Result<WorkerReport, WorkerError> {
    let capabilities = check_capabilities()?;

    let mut uninterrupted = engine(0..2)?;
    complete(&mut uninterrupted, EDGES[2])?;

    let mut worker = engine(0..2)?;
    let cancel = AtomicBool::new(false);
    let mut records = Vec::new();
    let cancelled = controlled(&mut worker, &cancel, &mut records)?;
    if cancelled.status != AdvanceStatus::Cancelled {
        return Err(format!(
            "expected cancellation after {CANCEL_AFTER} steps, got {:?}: lower CANCEL_AFTER",
            cancelled.status
        )
        .into());
    }
    let cancel_package = worker.snapshot().data()?;
    cancel.store(false, Ordering::SeqCst);
    let resumed = controlled(&mut worker, &cancel, &mut records)?;
    if resumed.status != AdvanceStatus::ReachedTarget {
        return Err(format!(
            "resumed call stopped as {:?}: raise MAX_STEPS",
            resumed.status
        )
        .into());
    }

    let mut restored = engine(0..2)?;
    restored.restore_data(&cancel_package)?;
    complete(&mut restored, EDGES[2])?;
    let mut resumed_package = Engine::resume(
        crate::runtime::client(),
        &cancel_package,
        cancel_package.setup.forcing.clone(),
    )?;
    complete(&mut resumed_package, EDGES[2])?;

    let at_boundary = [
        RunRecord::capture(&uninterrupted)?,
        RunRecord::capture(&worker)?,
        RunRecord::capture(&restored)?,
        RunRecord::capture(&resumed_package)?,
    ];

    let mut single = engine(0..3)?;
    complete(&mut single, EDGES[2])?;
    complete(&mut single, EDGES[3])?;
    let boundary_package = worker.snapshot().data()?;
    let mut forecast_resume =
        Engine::resume(crate::runtime::client(), &boundary_package, schedule(2..3)?)?;
    let mut forecasts = Vec::new();
    for (label, live) in [
        ("uninterrupted", &mut uninterrupted),
        ("worker", &mut worker),
        ("restored", &mut restored),
        ("resumed", &mut resumed_package),
    ] {
        live.continue_with_forcing(schedule(2..3)?)?;
        complete(live, EDGES[3])?;
        forecasts.push((label, RunRecord::capture(live)?));
    }
    complete(&mut forecast_resume, EDGES[3])?;
    forecasts.push(("resumed at boundary", RunRecord::capture(&forecast_resume)?));

    let budget = WaterBudget::new(worker.ledger(), worker.config());
    let [uninterrupted, worker, restored, resumed_package] = at_boundary;
    Ok(WorkerReport {
        capabilities,
        cancelled,
        cancel_package,
        resumed,
        records,
        uninterrupted,
        worker,
        restored,
        resumed_package,
        single_forecast: RunRecord::capture(&single)?,
        forecasts,
        budget,
    })
}
