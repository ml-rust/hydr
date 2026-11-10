//! Frozen terrain invariants and film sensitivity. Overtopping and film differences have no accuracy gate.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "support/impervious.rs"]
mod impervious;

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    AdvanceStatus, BoundaryConditions, ForcingInterval, ForcingSchedule, Solver, SolverConfig,
    StepDiagnostics,
};
use impervious::impervious_solver;

type TestResult = Result<(), Box<dyn std::error::Error>>;

const REST_DRIFT: f64 = 1e-11;
const FILM_DEPTHS: [f64; 3] = [0.0, 1e-8, 1e-6];

fn create(
    grid: CartesianGrid,
    bed: Vec<f64>,
    cells: Vec<CellState>,
    max_step: f64,
    film_depth: f64,
) -> Result<Solver<crate::runtime::Runtime>, Box<dyn std::error::Error>> {
    let count = grid.cell_count();
    let domain = Domain::new(
        grid,
        bed,
        vec![0.0; count],
        vec![true; count],
        SpatialReference::new(
            "synthetic projected metres".into(),
            "synthetic SI datum".into(),
            [0.0; 2],
        )?,
    )?;
    let state = FlowState::new(&domain, cells)?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 10.0,
            potential_evaporation: vec![0.0; count],
            rain: vec![0.0; count],
            withdrawal: vec![0.0; count],
            boundaries: vec![],
        }],
    )?;
    impervious_solver(
        domain,
        state,
        boundaries,
        forcing,
        SolverConfig {
            max_step,
            momentum_film_depth: film_depth,
            ..SolverConfig::default()
        },
        0.0,
    )
}

fn tolerance(scale: f64) -> f64 {
    let config = SolverConfig::default();
    config.volume_absolute + config.volume_relative * scale.max(1.0)
}

fn check_step(step: &StepDiagnostics) {
    let scale = step.storage_before.max(step.storage_after).max(1.0);
    assert!(step.residual.is_finite());
    assert!(step.residual.abs() <= tolerance(scale), "{step:?}");
    assert!(step.max_local_residual <= tolerance(scale), "{step:?}");
    assert!(step.momentum_removed.iter().all(|value| value.is_finite()));
    assert_eq!(step.rain_volume, 0.0);
    assert_eq!(step.withdrawal_volume, 0.0);
    assert_eq!(step.boundary_in, 0.0);
    assert_eq!(step.boundary_out, 0.0);
}

fn advance(solver: &mut Solver<crate::runtime::Runtime>, target: f64) -> TestResult {
    let report = solver.advance_controlled(
        target,
        100_000,
        || false,
        |step, cells| {
            check_step(step);
            assert!(cells.iter().all(|cell| cell.h.is_finite() && cell.h >= 0.0));
        },
    )?;
    assert_eq!(report.status, AdvanceStatus::ReachedTarget);
    let ledger = solver.ledger();
    let scale = ledger.initial_storage.max(ledger.storage).max(1.0);
    assert!(ledger.residual.is_finite());
    assert!(ledger.residual.abs() <= tolerance(scale), "{ledger:?}");
    assert!((ledger.storage - ledger.initial_storage).abs() <= tolerance(scale));
    assert_eq!(ledger.water_adjustment, 0.0);
    Ok(())
}

#[test]
fn stepped_shallow_lakes_preserve_rest_at_both_resolutions() -> TestResult {
    for refinement in [1, 2] {
        let (nx, ny) = (12 * refinement, 8 * refinement);
        let spacing = 1.0 / refinement as f64;
        let levels = [-2.0, 0.0, 1.0 - 1e-9, 1.0, 2.0];
        let bed: Vec<_> = (0..nx * ny)
            .map(|index| {
                let column = (index % nx) / refinement;
                let row = (index / nx) / refinement;
                levels[(column + 2 * row) % levels.len()]
            })
            .collect();
        let initial: Vec<_> = bed
            .iter()
            .map(|bed| CellState {
                h: f64::max(1.0 - bed, 0.0),
                qx: 0.0,
                qy: 0.0,
            })
            .collect();
        assert!(initial.iter().any(|cell| cell.h == 0.0));
        assert!(initial.iter().any(|cell| cell.h > 0.0 && cell.h < 1e-8));
        let mut solver = create(
            CartesianGrid::new(nx, ny, spacing, spacing)?,
            bed,
            initial.clone(),
            0.02,
            0.0,
        )?;
        advance(&mut solver, 0.2)?;
        let mut depth_drift = 0.0_f64;
        let mut momentum_drift = 0.0_f64;
        for (cell, initial) in solver.state()?.cells().iter().zip(initial) {
            depth_drift = depth_drift.max((cell.h - initial.h).abs());
            momentum_drift = momentum_drift.max(cell.qx.abs()).max(cell.qy.abs());
        }
        println!(
            "Stepped lake: nx={nx}, ny={ny}, t=0.2, max_depth_drift_m={depth_drift:.12e}, max_discharge_drift_m2_per_s={momentum_drift:.12e}"
        );
        assert!(depth_drift <= REST_DRIFT);
        assert!(momentum_drift <= REST_DRIFT);
    }
    Ok(())
}

#[test]
fn raised_crests_block_or_pass_water_by_free_surface_height() -> TestResult {
    let nx = 48;
    let dx = 0.25;
    for crest in [1.5, 0.2] {
        let bed = (0..nx)
            .map(|column| {
                if (16..20).contains(&column) {
                    crest
                } else {
                    0.0
                }
            })
            .collect();
        let initial = (0..nx)
            .map(|column| CellState {
                h: if column < 16 { 1.0 } else { 0.0 },
                qx: 0.0,
                qy: 0.0,
            })
            .collect();
        let mut solver = create(CartesianGrid::new(nx, 1, dx, 1.0)?, bed, initial, 0.01, 0.0)?;
        advance(&mut solver, 1.5)?;
        let downstream_storage = solver.state()?.cells()[20..]
            .iter()
            .map(|cell| cell.h * dx)
            .sum::<f64>();
        println!(
            "Crest invariant: crest_m={crest}, upstream_eta_m=1, t=1.5, downstream_storage_m3={downstream_storage:.12e}"
        );
        if crest > 1.0 {
            assert!(
                solver.state()?.cells()[16..]
                    .iter()
                    .all(|cell| cell.h == 0.0)
            );
            assert_eq!(downstream_storage, 0.0);
        } else {
            assert!(downstream_storage > 0.0);
        }
    }
    Ok(())
}

#[test]
fn shallow_wetting_reports_film_depth_sensitivity_without_accuracy_gate() -> TestResult {
    let nx = 64;
    let dx = 0.00025;
    let width = 1000.0;
    let mut baseline: Option<Vec<f64>> = None;
    for film_depth in FILM_DEPTHS {
        let cells = (0..nx)
            .map(|column| {
                let (h, qx) = if column < 16 {
                    (2e-6, 6e-9)
                } else if column < 32 {
                    (1e-10, 0.0)
                } else {
                    (0.0, 0.0)
                };
                CellState { h, qx, qy: 0.0 }
            })
            .collect();
        let mut solver = create(
            CartesianGrid::new(nx, 1, dx, width)?,
            vec![0.0; nx],
            cells,
            0.01,
            film_depth,
        )?;
        advance(&mut solver, 0.5)?;
        let depth: Vec<_> = solver.state()?.cells().iter().map(|cell| cell.h).collect();
        let mean_depth_l1 = baseline.as_ref().map_or(0.0, |baseline| {
            depth
                .iter()
                .zip(baseline)
                .map(|(h, base)| (h - base).abs())
                .sum::<f64>()
                / nx as f64
        });
        let integrated_depth_l1 = mean_depth_l1 * nx as f64 * dx * width;
        assert!(mean_depth_l1.is_finite());
        assert!(solver.ledger().storage > 0.0);
        assert!(
            solver.state()?.cells()[32..]
                .iter()
                .any(|cell| cell.h > 0.0)
        );
        println!(
            "Shallow wetting sensitivity: film_m={film_depth:.12e}, t=0.5, mean_depth_l1_m={mean_depth_l1:.12e}, integrated_depth_l1_m3={integrated_depth_l1:.12e}, storage_m3={:.12e}, water_residual_m3={:.12e}, signed_momentum_removed_m4_per_s={:?}",
            solver.ledger().storage,
            solver.ledger().residual,
            solver.ledger().momentum_removed,
        );
        if baseline.is_none() {
            assert_eq!(solver.ledger().momentum_removed, [0.0; 2]);
            baseline = Some(depth);
        }
    }
    Ok(())
}

#[test]
fn film_resets_report_signed_momentum_and_retain_positive_water() -> TestResult {
    for film_depth in FILM_DEPTHS {
        let mut solver = create(
            CartesianGrid::new(1, 1, 1.0, 1.0)?,
            vec![0.0],
            vec![CellState {
                h: 5e-9,
                qx: 1e-11,
                qy: -2e-11,
            }],
            1e-4,
            film_depth,
        )?;
        advance(&mut solver, 1e-4)?;
        let cell = solver.state()?.cells()[0];
        let step = solver.last_step()?.ok_or("missing accepted film step")?;
        assert!(cell.h > 0.0 && cell.h < 1e-8);
        if film_depth == 0.0 {
            assert!(cell.qx > 0.0 && cell.qy < 0.0);
            assert_eq!(step.momentum_removed, [0.0; 2]);
        } else {
            assert_eq!([cell.qx, cell.qy], [0.0; 2]);
            assert!(step.momentum_removed[0] > 0.0);
            assert!(step.momentum_removed[1] < 0.0);
            assert_eq!(solver.ledger().momentum_removed, step.momentum_removed);
        }
        println!(
            "Film reset: film_m={film_depth:.12e}, retained_h_m={:.12e}, signed_momentum_removed_m4_per_s={:?}",
            cell.h, step.momentum_removed,
        );
    }
    Ok(())
}
