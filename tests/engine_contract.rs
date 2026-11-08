//! Checks numerical execution, conservation, and complete accepted-state restoration.

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    AdvanceStatus, Axis, BoundaryConditions, BoundaryFace, CellResult, FluxPort, ForcingInterval,
    ForcingSchedule, Solver, SolverConfig,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn domain(
    nx: usize,
    ny: usize,
    bed: Vec<f64>,
    active: Vec<bool>,
) -> Result<Domain, Box<dyn std::error::Error>> {
    Ok(Domain::new(
        CartesianGrid::new(nx, ny, 1.0, 1.0)?,
        bed,
        vec![0.0; nx * ny],
        active,
        SpatialReference::new("EPSG:3375".into(), "synthetic SI datum".into(), [0.0; 2])?,
    )?)
}

fn interval(start: f64, end: f64, rain: Vec<f64>, withdrawal: Vec<f64>) -> ForcingInterval {
    ForcingInterval {
        start,
        end,
        rain,
        withdrawal,
        ports: Vec::new(),
    }
}

fn walls_solver(
    domain: Domain,
    cells: Vec<CellState>,
    intervals: Vec<ForcingInterval>,
) -> Result<Solver, Box<dyn std::error::Error>> {
    let state = FlowState::new(&domain, cells)?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(&domain, &boundaries, intervals)?;
    Ok(Solver::new(
        domain,
        state,
        boundaries,
        forcing,
        SolverConfig::default(),
        0.0,
    )?)
}

fn resting(depth: f64) -> CellState {
    CellState {
        h: depth,
        ..CellState::default()
    }
}

#[test]
fn irregular_lake_preserves_rest_and_partial_dry_cells() -> TestResult {
    let count = 24 * 16;
    let bed: Vec<_> = (0..count)
        .map(|index| match (index % 24 + index / 24) % 4 {
            0 => 0.0,
            1 => 0.3,
            2 => 1.2,
            _ => 0.8,
        })
        .collect();
    let cells: Vec<_> = bed
        .iter()
        .map(|z| resting(f64::max(1.0 - z, 0.0)))
        .collect();
    let domain = domain(24, 16, bed, vec![true; count])?;
    let mut solver = walls_solver(
        domain,
        cells.clone(),
        vec![interval(0.0, 11.0, vec![0.0; count], vec![0.0; count])],
    )?;
    assert_eq!(
        solver.advance_to(10.0, 10_000)?.status,
        AdvanceStatus::ReachedTarget
    );
    for (actual, initial) in solver.state().cells().iter().zip(&cells) {
        assert!(
            (actual.h - initial.h).abs() <= 1e-11,
            "{actual:?} versus {initial:?}"
        );
        assert!(actual.qx.abs() <= 1e-11, "{actual:?}");
        assert!(actual.qy.abs() <= 1e-11, "{actual:?}");
    }
    assert!(solver.ledger().residual.abs() <= 1e-9);
    Ok(())
}

#[test]
fn rainfall_steps_stop_at_interval_edges_and_account_for_storage() -> TestResult {
    let domain = domain(2, 2, vec![0.0; 4], vec![true; 4])?;
    let mut solver = walls_solver(
        domain,
        vec![resting(0.0); 4],
        vec![
            interval(0.0, 0.3, vec![0.01; 4], vec![0.0; 4]),
            interval(0.3, 2.0, vec![0.02; 4], vec![0.0; 4]),
        ],
    )?;
    let mut accepted_ends = Vec::new();
    let report = solver.advance_controlled(
        1.0,
        100,
        || false,
        |step, _| {
            accepted_ends.push(step.end);
            assert!(step.normalized_residual.abs() <= 1.0);
        },
    )?;
    assert_eq!(report.status, AdvanceStatus::ReachedTarget);
    assert!(accepted_ends.contains(&0.3));
    let expected = 4.0 * (0.01 * 0.3 + 0.02 * 0.7);
    assert!((solver.ledger().rain - expected).abs() <= 1e-12);
    assert!((solver.ledger().storage - expected).abs() <= 1e-12);
    assert_eq!(solver.ledger().water_adjustment, 0.0);
    Ok(())
}

#[test]
fn repeated_runs_and_restart_preserve_exact_accepted_state() -> TestResult {
    fn create() -> Result<Solver, Box<dyn std::error::Error>> {
        walls_solver(
            domain(3, 2, vec![0.0; 6], vec![true; 6])?,
            vec![resting(0.2); 6],
            vec![interval(0.0, 3.0, vec![0.001; 6], vec![0.0; 6])],
        )
    }
    let mut first = create()?;
    let mut repeated = create()?;
    first.advance_to(0.5, 100)?;
    repeated.advance_to(0.5, 100)?;
    assert_eq!(first.snapshot(), repeated.snapshot());
    let midpoint = first.snapshot();
    let mut restarted = create()?;
    restarted.restore(&midpoint)?;
    first.advance_to(1.0, 100)?;
    repeated.advance_to(1.0, 100)?;
    restarted.advance_to(1.0, 100)?;
    assert_eq!(first.snapshot(), repeated.snapshot());
    assert_eq!(first.snapshot(), restarted.snapshot());
    assert_eq!(first.last_step(), restarted.last_step());
    assert_eq!(first.ledger(), restarted.ledger());
    Ok(())
}

#[test]
fn rejected_withdrawal_and_invalid_target_preserve_snapshot() -> TestResult {
    let mut solver = walls_solver(
        domain(1, 1, vec![0.0], vec![true])?,
        vec![resting(0.0)],
        vec![interval(0.0, 2.0, vec![0.0], vec![0.1])],
    )?;
    let before = solver.snapshot();
    assert!(solver.advance_to(0.1, 100).is_err());
    assert_eq!(solver.snapshot(), before);
    assert!(solver.advance_to(f64::NAN, 100).is_err());
    assert_eq!(solver.snapshot(), before);
    assert!(solver.advance_to(-1.0, 100).is_err());
    assert_eq!(solver.snapshot(), before);
    Ok(())
}

#[test]
fn incompatible_restore_preserves_destination_snapshot() -> TestResult {
    let forcing = vec![interval(0.0, 2.0, vec![0.0], vec![0.0])];
    let source = walls_solver(
        domain(1, 1, vec![0.0], vec![true])?,
        vec![resting(1.0)],
        forcing.clone(),
    )?;
    let mut destination = walls_solver(
        domain(1, 1, vec![0.5], vec![true])?,
        vec![resting(1.0)],
        forcing,
    )?;
    let before = destination.snapshot();
    assert!(destination.restore(&source.snapshot()).is_err());
    assert_eq!(destination.snapshot(), before);
    Ok(())
}

#[test]
fn cancellation_and_step_limits_return_accepted_progress() -> TestResult {
    let mut solver = walls_solver(
        domain(1, 1, vec![0.0], vec![true])?,
        vec![resting(1.0)],
        vec![interval(0.0, 3.0, vec![0.0], vec![0.0])],
    )?;
    let before = solver.snapshot();
    let mut callbacks = 0;
    let report = solver.advance_controlled(1.0, 100, || true, |_, _| callbacks += 1)?;
    assert_eq!(report.status, AdvanceStatus::Cancelled);
    assert_eq!(report.accepted_steps, 0);
    assert_eq!(callbacks, 0);
    assert_eq!(solver.snapshot(), before);
    let report = solver.advance_controlled(
        1.0,
        1,
        || false,
        |step, cells| {
            callbacks += 1;
            assert_eq!(step.end - step.start, step.dt);
            assert_eq!(cells.len(), 1);
        },
    )?;
    assert_eq!(report.status, AdvanceStatus::StepLimit);
    assert_eq!(report.accepted_steps, 1);
    assert_eq!(callbacks, 1);
    assert_eq!(report.accepted_time, solver.time());
    assert!(solver.time() > 0.0 && solver.time() < 1.0);
    assert_eq!(
        solver.advance_to(1.0, 100)?.status,
        AdvanceStatus::ReachedTarget
    );
    Ok(())
}

#[test]
fn mask_barriers_isolate_storage_and_rainfall() -> TestResult {
    let mut solver = walls_solver(
        domain(
            5,
            1,
            vec![0.0, 0.0, f64::NAN, 0.0, 0.0],
            vec![true, true, false, true, true],
        )?,
        vec![
            resting(0.5),
            resting(0.5),
            resting(0.0),
            resting(0.0),
            resting(0.0),
        ],
        vec![interval(
            0.0,
            2.0,
            vec![0.01, 0.01, 0.0, 0.0, 0.0],
            vec![0.0; 5],
        )],
    )?;
    solver.advance_to(1.0, 100)?;
    assert_eq!(&solver.state().cells()[2..], &[resting(0.0); 3]);
    assert_eq!(solver.cell(2), CellResult::Unsupported);
    assert_eq!(solver.cell(3), CellResult::Dry);
    assert!((solver.ledger().rain - 0.02).abs() <= 1e-12);
    assert!((solver.ledger().storage - 1.02).abs() <= 1e-12);
    Ok(())
}

#[test]
fn outputs_distinguish_wet_dry_and_unsupported_locations() -> TestResult {
    let solver = walls_solver(
        domain(3, 1, vec![2.0, 0.0, f64::NAN], vec![true, true, false])?,
        vec![
            CellState {
                h: 1.0,
                qx: 2.0,
                qy: -3.0,
            },
            resting(0.0),
            resting(0.0),
        ],
        vec![interval(0.0, 1.0, vec![0.0; 3], vec![0.0; 3])],
    )?;
    assert_eq!(
        solver.cell(0),
        CellResult::Wet {
            depth: 1.0,
            elevation: 3.0,
            velocity: [2.0, -3.0],
            discharge_per_width: [2.0, -3.0],
        }
    );
    assert_eq!(solver.cell(1), CellResult::Dry);
    assert_eq!(solver.cell(2), CellResult::Unsupported);
    assert_eq!(solver.cell(3), CellResult::Unsupported);
    assert_eq!(solver.cell_at(0.5, 0.5), solver.cell(0));
    assert_eq!(solver.cell_at(2.5, 0.5), CellResult::Unsupported);
    assert_eq!(solver.cell_at(-0.1, 0.5), CellResult::Unsupported);
    assert_eq!(solver.cell_at(3.0, 0.5), CellResult::Unsupported);
    assert_eq!(solver.cell_at(f64::NAN, 0.5), CellResult::Unsupported);
    Ok(())
}

#[test]
fn boundaries_reject_interior_duplicate_and_inactive_faces() -> TestResult {
    let domain = domain(2, 1, vec![0.0; 2], vec![true, false])?;
    let left = BoundaryFace {
        axis: Axis::X,
        column: 0,
        row: 0,
    };
    assert!(BoundaryConditions::with_ports(&domain, vec![left, left]).is_err());
    for face in [
        BoundaryFace {
            axis: Axis::X,
            column: 1,
            row: 0,
        },
        BoundaryFace {
            axis: Axis::X,
            column: 2,
            row: 0,
        },
        BoundaryFace {
            axis: Axis::Y,
            column: 2,
            row: 0,
        },
    ] {
        assert!(BoundaryConditions::with_ports(&domain, vec![face]).is_err());
    }
    let boundaries = BoundaryConditions::with_ports(&domain, vec![left])?;
    assert!(
        ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![interval(0.0, 1.0, vec![0.0; 2], vec![0.0; 2]),]
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn forcing_rejects_gaps_invalid_rates_and_inactive_support() -> TestResult {
    let domain = domain(2, 1, vec![0.0; 2], vec![true, false])?;
    let boundaries = BoundaryConditions::impermeable_walls();
    for intervals in [
        vec![],
        vec![interval(0.0, 1.0, vec![-0.1, 0.0], vec![0.0; 2])],
        vec![interval(0.0, 1.0, vec![0.0, 0.1], vec![0.0; 2])],
        vec![interval(0.0, 1.0, vec![0.0; 2], vec![f64::NAN, 0.0])],
        vec![interval(0.0, 1.0, vec![0.0], vec![0.0; 2])],
        vec![
            interval(0.0, 0.3, vec![0.0; 2], vec![0.0; 2]),
            interval(0.4, 1.0, vec![0.0; 2], vec![0.0; 2]),
        ],
    ] {
        assert!(ForcingSchedule::new(&domain, &boundaries, intervals).is_err());
    }
    Ok(())
}

#[test]
fn manufactured_ports_account_for_inward_and_outward_volume() -> TestResult {
    for water_rate in [-0.01_f64, 0.01] {
        let domain = domain(1, 1, vec![0.0], vec![true])?;
        let state = FlowState::new(&domain, vec![resting(1.0)])?;
        let face = BoundaryFace {
            axis: Axis::X,
            column: 0,
            row: 0,
        };
        let boundaries = BoundaryConditions::with_ports(&domain, vec![face])?;
        let mut forcing_interval = interval(0.0, 1.0, vec![0.0], vec![0.0]);
        forcing_interval.ports.push(FluxPort {
            face,
            water_rate,
            momentum_rate: [-4.905, 0.0],
            signal_speed: 10.0,
            provenance: "Manufactured full flux, h near 1 m, speed bound 10 m/s".into(),
        });
        let forcing = ForcingSchedule::new(&domain, &boundaries, vec![forcing_interval])?;
        let mut solver = Solver::new(
            domain,
            state,
            boundaries,
            forcing,
            SolverConfig::default(),
            0.0,
        )?;
        solver.advance_to(0.1, 100)?;
        assert!((solver.ledger().boundary_in - (-water_rate).max(0.0) * 0.1).abs() <= 1e-12);
        assert!((solver.ledger().boundary_out - water_rate.max(0.0) * 0.1).abs() <= 1e-12);
        assert!((solver.ledger().storage - (1.0 - water_rate * 0.1)).abs() <= 1e-12);
        assert!(solver.ledger().residual.abs() <= 1e-9);
    }
    Ok(())
}
