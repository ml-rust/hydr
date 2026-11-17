//! Atomic rejection of continuation forcing that rewrites or misses accepted history.

use super::continuation::{
    EDGES, Engine, TestResult, engine, interval, other_future, reference, schedule,
};
use hydr::solver::{BoundaryConditions, BoundaryForcing, ForcingInterval, ForcingSchedule};

/// Asserts that every candidate rejects and leaves the package unchanged.
fn assert_rejected(
    solver: &mut Engine,
    candidates: Vec<(&str, ForcingSchedule)>,
    expected: &str,
) -> TestResult {
    let before = solver.snapshot().data()?;
    for (name, forcing) in candidates {
        let error = solver
            .continue_with_forcing(forcing)
            .err()
            .ok_or_else(|| format!("{name}: continuation must reject"))?;
        assert!(error.to_string().contains(expected), "{name}: {error}");
        assert_eq!(solver.snapshot().data()?, before, "{name}");
    }
    Ok(())
}

fn flip(value: f64) -> f64 {
    f64::from_bits(value.to_bits() ^ 1)
}

fn rename(interval: &mut ForcingInterval, provenance: &str) {
    for boundary in &mut interval.boundaries {
        if let BoundaryForcing::Flux(port) = boundary {
            port.provenance = provenance.into();
        }
    }
}

#[test]
fn continuation_rejects_rewritten_past_forcing() -> TestResult {
    let mut at_boundary = engine(reference(0..2), EDGES[2])?;
    let mut flipped = reference(0..3);
    flipped[0].rain[0] = flip(flipped[0].rain[0]);
    let mut signed = reference(0..3);
    signed[1].potential_evaporation[1] = -0.0;
    let mut shortened = reference(0..3);
    shortened[1].end = 0.55;
    shortened[2].start = 0.55;
    let mut renamed = reference(0..3);
    rename(&mut renamed[1], "rewritten gauge");
    assert_rejected(
        &mut at_boundary,
        vec![
            ("flipped rain bit", schedule(flipped)?),
            ("negative zero", schedule(signed)?),
            ("changed frozen end", schedule(shortened)?),
            ("changed provenance", schedule(renamed)?),
        ],
        "frozen interval",
    )?;
    assert_rejected(
        &mut at_boundary,
        vec![(
            "start between boundaries",
            schedule(vec![interval(0.5, EDGES[3], 2.0)])?,
        )],
        "retained interval start",
    )?;

    let mut inside = engine(reference(0..3), 0.45)?;
    assert_rejected(
        &mut inside,
        vec![
            (
                "replaces current interval",
                schedule(vec![interval(EDGES[1], EDGES[3], 5.0)])?,
            ),
            (
                "splits current interval at t",
                schedule(vec![
                    interval(EDGES[1], 0.45, 2.0),
                    interval(0.45, EDGES[3], 2.0),
                ])?,
            ),
        ],
        "frozen interval",
    )?;
    assert_rejected(
        &mut inside,
        vec![(
            "starts at t inside interval",
            schedule(vec![interval(0.45, EDGES[3], 2.0)])?,
        )],
        "retained interval start",
    )?;

    let mut folded = engine(reference(0..2), EDGES[2])?;
    folded.continue_with_forcing(schedule(reference(2..3))?)?;
    assert_eq!(folded.snapshot().data()?.forcing_prefix.intervals, 1);
    let mut altered = reference(0..3);
    altered[0].rain[1] = 9e-4;
    let mut unaligned = reference(0..3);
    unaligned[0].end = 0.2;
    unaligned.insert(1, interval(0.2, EDGES[1], 1.0));
    assert_rejected(
        &mut folded,
        vec![
            ("altered dropped prefix", schedule(altered)?),
            ("prefix split differently", schedule(unaligned)?),
        ],
        "folded prefix",
    )?;
    folded.continue_with_forcing(schedule(reference(0..3))?)?;
    Ok(())
}

#[test]
fn continuation_rejects_uncovered_or_incompatible_schedule() -> TestResult {
    let mut solver = engine(reference(0..2), EDGES[2])?;
    assert_rejected(
        &mut solver,
        vec![("ends before t", schedule(reference(0..1))?)],
        "missing",
    )?;
    assert_rejected(
        &mut solver,
        vec![(
            "starts after t",
            schedule(vec![interval(0.7, EDGES[3], 3.0)])?,
        )],
        "at or before the accepted time",
    )?;
    let mut walls = interval(EDGES[2], EDGES[3], 3.0);
    walls.boundaries.clear();
    let domain = solver.domain().clone();
    let wrong_faces = ForcingSchedule::new(
        &domain,
        &BoundaryConditions::impermeable_walls(),
        vec![walls],
    )?;
    assert_rejected(
        &mut solver,
        vec![("wrong open-face set", wrong_faces)],
        "declared open face",
    )
}

#[test]
fn restore_rejects_snapshot_outside_destination_window_or_other_future() -> TestResult {
    let mut solver = engine(reference(0..2), 0.2)?;
    let early = solver.snapshot();
    let early_data = early.data()?;
    solver.advance_to(EDGES[2], 10_000)?;
    solver.continue_with_forcing(schedule(reference(2..3))?)?;
    let other = engine(other_future(), EDGES[2])?;
    let before = solver.snapshot().data()?;
    for (name, result) in [
        ("early snapshot", solver.restore(&early)),
        ("early package", solver.restore_data(&early_data)),
        ("other future snapshot", solver.restore(&other.snapshot())),
        (
            "other future package",
            solver.restore_data(&other.snapshot().data()?),
        ),
    ] {
        assert!(result.is_err(), "{name} must reject");
        assert_eq!(solver.snapshot().data()?, before, "{name}");
    }
    let error = solver
        .restore(&early)
        .err()
        .ok_or("early snapshot must reject")?;
    assert!(
        error.to_string().contains("starts at interval 1"),
        "{error}"
    );
    Ok(())
}
