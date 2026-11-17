//! Bounded forcing windows, folded prefixes, and their restart checks.

use super::impervious::impervious_solver;
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, SNAPSHOT_SCHEMA, SnapshotData, Solver,
    SolverConfig,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;
type Engine = Solver<crate::runtime::Runtime>;

const INTERVALS: usize = 50;
const WIDTH: f64 = 0.125;

/// Returns the exact start of interval `k`.
fn edge(k: usize) -> f64 {
    k as f64 * WIDTH
}

fn create(depth: f64) -> Result<Engine, Box<dyn std::error::Error>> {
    let domain = Domain::new(
        CartesianGrid::new(2, 1, 1.0, 2.0)?,
        vec![0.0, 0.01],
        vec![0.03; 2],
        vec![true; 2],
        SpatialReference::new("EPSG:3375".into(), "synthetic SI datum".into(), [0.0; 2])?,
    )?;
    let state = FlowState::new(
        &domain,
        vec![
            CellState {
                h: depth,
                qx: 0.0,
                qy: 0.0
            };
            2
        ],
    )?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let intervals = (0..INTERVALS)
        .map(|k| ForcingInterval {
            start: edge(k),
            end: edge(k + 1),
            potential_evaporation: vec![0.0; 2],
            rain: vec![1e-4 * (1 + k % 3) as f64, 2e-4 * (1 + k % 5) as f64],
            withdrawal: vec![0.0; 2],
            boundaries: vec![],
        })
        .collect();
    let forcing = ForcingSchedule::new(&domain, &boundaries, intervals)?;
    impervious_solver(
        domain,
        state,
        boundaries,
        forcing,
        SolverConfig::default(),
        0.0,
    )
}

/// Advances a fresh source engine to the end of 1-based interval 40 (index 39).
fn at_interval_40_end() -> Result<Engine, Box<dyn std::error::Error>> {
    let mut source = create(1.0)?;
    source.advance_to(edge(40), 1000)?;
    Ok(source)
}

fn assert_same_state(left: &Engine, right: &Engine) -> TestResult {
    assert_eq!(left.time().to_bits(), right.time().to_bits());
    for (left, right) in left.state()?.cells().iter().zip(right.state()?.cells()) {
        assert_eq!(left.h.to_bits(), right.h.to_bits());
        assert_eq!(left.qx.to_bits(), right.qx.to_bits());
        assert_eq!(left.qy.to_bits(), right.qy.to_bits());
    }
    assert_eq!(left.snapshot().data()?, right.snapshot().data()?);
    Ok(())
}

fn assert_rejected(packages: &[SnapshotData]) -> TestResult {
    let mut destination = create(0.2)?;
    destination.advance_to(0.02, 100)?;
    let before = destination.snapshot().data()?;
    for (index, data) in packages.iter().enumerate() {
        assert!(destination.restore_data(data).is_err(), "forgery {index}");
        assert_eq!(destination.snapshot().data()?, before, "forgery {index}");
    }
    Ok(())
}

#[test]
fn package_forcing_window_is_bounded() -> TestResult {
    let fresh = create(1.0)?.snapshot().data()?;
    assert_eq!(fresh.setup.forcing.intervals().len(), INTERVALS);
    assert_eq!(fresh.forcing_prefix.intervals, 0);
    assert_eq!(fresh.forcing_prefix.start.to_bits(), 0.0_f64.to_bits());
    assert_eq!(fresh.forcing_prefix.end.to_bits(), 0.0_f64.to_bits());

    // The last step starts inside index 39, so rule W retains 39.. and folds 0..39.
    let package = at_interval_40_end()?.snapshot().data()?;
    let step = package.last_step.as_ref().ok_or("missing accepted step")?;
    assert!(edge(39) <= step.start && step.end == edge(40));
    let window = package.setup.forcing.intervals();
    assert_eq!(window.len(), 1 + (INTERVALS - 40));
    assert_eq!(window[0].start.to_bits(), edge(39).to_bits());
    assert_eq!(package.forcing_prefix.intervals, 39);
    assert_eq!(package.forcing_prefix.start.to_bits(), 0.0_f64.to_bits());
    assert_eq!(package.forcing_prefix.end.to_bits(), edge(39).to_bits());

    let mut late = create(1.0)?;
    late.advance_to(edge(INTERVALS), 10_000)?;
    let last = late.snapshot().data()?;
    assert_eq!(last.setup.forcing.intervals().len(), 1);
    assert_eq!(last.forcing_prefix.intervals, INTERVALS - 1);
    Ok(())
}

#[test]
fn compacted_package_restores_into_full_schedule_engine() -> TestResult {
    let mut original = at_interval_40_end()?;
    let snapshot = original.snapshot();
    let package = snapshot.data()?;
    let mut imported = create(0.0)?;
    let mut trusted = create(0.0)?;
    imported.restore_data(&package)?;
    trusted.restore(&snapshot)?;
    assert_eq!(imported.snapshot().data()?, package);
    assert_eq!(trusted.snapshot().data()?, package);
    for target in [edge(40) + 0.05, edge(43), edge(47) + 0.01, edge(INTERVALS)] {
        original.advance_to(target, 10_000)?;
        imported.advance_to(target, 10_000)?;
        trusted.advance_to(target, 10_000)?;
        assert_same_state(&original, &imported)?;
        assert_same_state(&original, &trusted)?;
    }
    Ok(())
}

#[test]
fn package_rejects_forged_forcing_prefix() -> TestResult {
    let good = at_interval_40_end()?.snapshot().data()?;
    let mut later = at_interval_40_end()?;
    later.advance_to(edge(40) + 0.5 * WIDTH, 1000)?;
    let later = later.snapshot().data()?;
    assert_eq!(later.forcing_prefix.intervals, 40);
    let fresh = create(1.0)?.snapshot().data()?;
    let mut forged = Vec::new();
    let mut add = |base: &SnapshotData, modify: &dyn Fn(&mut SnapshotData)| {
        let mut data = base.clone();
        modify(&mut data);
        forged.push(data);
    };
    add(&good, &|data| data.forcing_prefix.digest[7] ^= 1);
    add(&good, &|data| data.forcing_prefix.intervals = 38);
    add(&good, &|data| data.forcing_prefix.intervals = 40);
    add(&good, &|data| data.forcing_prefix.intervals = 0);
    add(&good, &|data| {
        data.forcing_prefix.end = f64::from_bits(data.forcing_prefix.end.to_bits() + 1);
    });
    add(&good, &|data| data.forcing_prefix.start = 1e-300);
    // A self-consistent window that starts after the last step interval.
    add(&good, &|data| {
        data.setup.forcing = later.setup.forcing.clone();
        data.forcing_prefix = later.forcing_prefix.clone();
    });
    add(&fresh, &|data| data.forcing_prefix.digest[0] ^= 1);
    add(&fresh, &|data| data.forcing_prefix.end = 1e-300);
    assert_rejected(&forged)?;
    for unforged in [&good, &later, &fresh] {
        create(0.2)?.restore_data(unforged)?;
    }
    Ok(())
}

#[test]
fn old_schema_package_rejects() -> TestResult {
    let mut old = at_interval_40_end()?.snapshot().data()?;
    assert_eq!(SNAPSHOT_SCHEMA, "hydr-checkpoint-2");
    old.schema = "hydr-checkpoint-1".into();
    assert_rejected(&[old])
}
