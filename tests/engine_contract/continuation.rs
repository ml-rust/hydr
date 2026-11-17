//! Forecast continuation under forcing spliced after the accepted history.

use crate::impervious::impervious_solver;
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    Axis, BoundaryConditions, BoundaryFace, FluxPort, ForcingInterval, ForcingSchedule,
    SnapshotData, Solver, SolverConfig, SolverSetup,
};

pub(super) type TestResult = Result<(), Box<dyn std::error::Error>>;
pub(super) type Engine = Solver<crate::runtime::Runtime>;

/// Interval edges of the reference schedule.
pub(super) const EDGES: [f64; 4] = [0.0, 0.3, 0.6, 1.0];

fn domain() -> Result<Domain, Box<dyn std::error::Error>> {
    Ok(Domain::new(
        CartesianGrid::new(2, 1, 1.0, 1.0)?,
        vec![0.0, 0.01],
        vec![0.03; 2],
        vec![true; 2],
        SpatialReference::new("EPSG:3375".into(), "synthetic SI datum".into(), [0.0; 2])?,
    )?)
}

const FACE: BoundaryFace = BoundaryFace {
    axis: Axis::X,
    column: 0,
    row: 0,
};

pub(super) fn port_boundaries(
    domain: &Domain,
) -> Result<BoundaryConditions, Box<dyn std::error::Error>> {
    Ok(BoundaryConditions::with_open_faces(domain, vec![FACE])?)
}

/// Returns interval `[start, end)` with rates scaled by `scale` and one inflow port.
pub(super) fn interval(start: f64, end: f64, scale: f64) -> ForcingInterval {
    ForcingInterval {
        start,
        end,
        rain: vec![1e-4 * scale, 2e-4 * scale],
        potential_evaporation: vec![0.0; 2],
        withdrawal: vec![0.0; 2],
        boundaries: vec![
            FluxPort {
                face: FACE,
                water_rate: -1e-3 * scale,
                momentum_rate: [-4.905, 0.0],
                signal_speed: 10.0,
                provenance: "manufactured inflow".into(),
            }
            .into(),
        ],
    }
}

/// Returns reference intervals `range` with distinct rates.
pub(super) fn reference(range: std::ops::Range<usize>) -> Vec<ForcingInterval> {
    range
        .map(|k| interval(EDGES[k], EDGES[k + 1], (k + 1) as f64))
        .collect()
}

/// Returns the reference schedule with interval 2 replaced by different rates.
pub(super) fn other_future() -> Vec<ForcingInterval> {
    let mut intervals = reference(0..2);
    intervals.push(interval(EDGES[2], EDGES[3], 7.0));
    intervals
}

pub(super) fn schedule(
    intervals: Vec<ForcingInterval>,
) -> Result<ForcingSchedule, Box<dyn std::error::Error>> {
    let domain = domain()?;
    let boundaries = port_boundaries(&domain)?;
    Ok(ForcingSchedule::new(&domain, &boundaries, intervals)?)
}

/// Builds an engine at time zero, then advances it to `time`.
pub(super) fn engine(
    intervals: Vec<ForcingInterval>,
    time: f64,
) -> Result<Engine, Box<dyn std::error::Error>> {
    let domain = domain()?;
    let boundaries = port_boundaries(&domain)?;
    let state = FlowState::new(
        &domain,
        vec![
            CellState {
                h: 1.0,
                qx: 0.0,
                qy: 0.0
            };
            2
        ],
    )?;
    let forcing = ForcingSchedule::new(&domain, &boundaries, intervals)?;
    let mut solver = impervious_solver(
        domain,
        state,
        boundaries,
        forcing,
        SolverConfig::default(),
        0.0,
    )?;
    solver.advance_to(time, 10_000)?;
    Ok(solver)
}

/// Asserts bitwise equality of package, ledger, last step, and step count.
pub(super) fn assert_same(left: &Engine, right: &Engine) -> TestResult {
    let (a, b) = (left.snapshot().data()?, right.snapshot().data()?);
    assert_eq!(a, b);
    assert_eq!(a.accepted.time.to_bits(), b.accepted.time.to_bits());
    for (x, y) in a.accepted.flow.cells().iter().zip(b.accepted.flow.cells()) {
        assert_eq!(
            [x.h, x.qx, x.qy].map(f64::to_bits),
            [y.h, y.qx, y.qy].map(f64::to_bits)
        );
    }
    assert_eq!(left.ledger(), right.ledger());
    assert_eq!(
        left.ledger().storage.to_bits(),
        right.ledger().storage.to_bits()
    );
    assert_eq!(left.last_step()?, right.last_step()?);
    assert_eq!(a.accepted_steps, b.accepted_steps);
    Ok(())
}

fn advance_all(engines: &mut [&mut Engine], target: f64) -> TestResult {
    for engine in engines {
        engine.advance_to(target, 10_000)?;
    }
    Ok(())
}

#[test]
fn extended_forcing_continuation_matches_single_schedule_bits() -> TestResult {
    let mut single = engine(reference(0..3), EDGES[2])?;
    let mut tail = engine(reference(0..2), EDGES[2])?;
    let package = tail.snapshot().data()?;
    tail.continue_with_forcing(schedule(reference(2..3))?)?;
    let mut full = engine(reference(0..2), EDGES[2])?;
    full.continue_with_forcing(schedule(reference(0..3))?)?;
    let mut resumed = Engine::resume(
        crate::runtime::client(),
        &package,
        schedule(reference(2..3))?,
    )?;
    for other in [&tail, &full, &resumed] {
        assert_same(&single, other)?;
    }
    advance_all(
        &mut [&mut single, &mut tail, &mut full, &mut resumed],
        EDGES[3],
    )?;
    for other in [&tail, &full, &resumed] {
        assert_same(&single, other)?;
    }
    Ok(())
}

#[test]
fn continuation_on_boundary_replaces_next_interval() -> TestResult {
    let mut continued = engine(reference(0..3), EDGES[2])?;
    continued.continue_with_forcing(schedule(other_future()[2..].to_vec())?)?;
    let window = continued.snapshot().data()?.setup.forcing;
    assert_eq!(window.intervals().last(), other_future().last());
    let mut single = engine(other_future(), EDGES[2])?;
    assert_same(&single, &continued)?;
    advance_all(&mut [&mut single, &mut continued], EDGES[3])?;
    assert_same(&single, &continued)
}

#[test]
fn continuation_inside_interval_keeps_current_interval() -> TestResult {
    let mut continued = engine(reference(0..3), 0.45)?;
    continued.continue_with_forcing(schedule(other_future()[1..].to_vec())?)?;
    let mut single = engine(other_future(), 0.45)?;
    assert_same(&single, &continued)?;
    advance_all(&mut [&mut single, &mut continued], EDGES[3])?;
    assert_same(&single, &continued)
}

#[test]
fn continuation_preserves_ledger_budget_steps_and_last_step() -> TestResult {
    let mut solver = engine(reference(0..3), 0.45)?;
    let before = solver.snapshot().data()?;
    let ledger = solver.ledger().clone();
    let last = solver.last_step()?;
    solver.continue_with_forcing(schedule(other_future()[1..].to_vec())?)?;
    let after = solver.snapshot().data()?;
    assert_eq!(after.accepted, before.accepted);
    assert_eq!(
        after.accepted.time.to_bits(),
        before.accepted.time.to_bits()
    );
    assert_eq!(after.ledger, ledger);
    assert_eq!(
        after.ledger.water_adjustment_abs_bound.to_bits(),
        ledger.water_adjustment_abs_bound.to_bits()
    );
    assert_eq!(after.setup.config, before.setup.config);
    assert_eq!(after.accepted_steps, before.accepted_steps);
    assert!(after.accepted_steps > 0);
    assert_eq!(solver.last_step()?, last);
    assert_eq!(after.last_step, before.last_step);
    assert_eq!(after.forcing_prefix, before.forcing_prefix);
    assert_ne!(after.setup.forcing, before.setup.forcing);
    Ok(())
}

#[test]
fn resume_with_package_forcing_equals_restore_data() -> TestResult {
    let package = engine(reference(0..3), 0.45)?.snapshot().data()?;
    let mut resumed = Engine::resume(
        crate::runtime::client(),
        &package,
        package.setup.forcing.clone(),
    )?;
    let mut restored = engine(reference(0..3), 0.0)?;
    restored.restore_data(&package)?;
    assert_eq!(resumed.snapshot().data()?, package);
    assert_same(&resumed, &restored)?;
    advance_all(&mut [&mut resumed, &mut restored], EDGES[3])?;
    assert_same(&resumed, &restored)
}

#[test]
fn new_epoch_from_accepted_state_resets_ledgers() -> TestResult {
    let previous = engine(reference(0..3), EDGES[2])?;
    let data: SnapshotData = previous.snapshot().data()?;
    let forcing = schedule(vec![interval(EDGES[2], 1.2, 4.0)])?;
    let mut epoch = Engine::new(
        crate::runtime::client(),
        SolverSetup {
            forcing,
            ..data.setup
        },
        data.accepted.clone(),
    )?;
    let fresh = epoch.snapshot().data()?;
    assert_eq!(fresh.accepted, data.accepted);
    assert_eq!(fresh.accepted_steps, 0);
    assert_eq!(fresh.last_step, None);
    assert_eq!(fresh.forcing_prefix.intervals, 0);
    assert_eq!(
        epoch.ledger().storage.to_bits(),
        previous.ledger().storage.to_bits()
    );
    assert_eq!(epoch.ledger().initial_storage, epoch.ledger().storage);
    assert_eq!(epoch.ledger().rain, 0.0);
    assert_eq!(epoch.ledger().boundary_in, 0.0);
    epoch.advance_to(1.2, 10_000)?;
    assert!(epoch.ledger().rain > 0.0);
    Ok(())
}
