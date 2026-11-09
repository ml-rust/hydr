//! Exact affine radial flow in a parabolic bowl, including its moving dry boundary.
//!
//! Initial and reference fields use cell-centre point samples, rather than exact cell averages.
//! The endpoint at 0.5 s checks contraction. Expansion, reversal, and full cycles remain unchecked.

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    AdvanceStatus, BoundaryConditions, ForcingInterval, ForcingSchedule, Solver, SolverConfig,
};

const G: f64 = 9.81;
const B: f64 = 0.1;
const H: f64 = 0.1;

fn exact(x: f64, y: f64, time: f64) -> CellState {
    let omega = (2.0 * G * B).sqrt();
    let (sin, cos) = (omega * time).sin_cos();
    let initial: f64 = 1.2;
    let a2 = initial.powi(2) * cos.powi(2) + initial.powi(-2) * sin.powi(2);
    let strain = omega * sin * cos * (initial.powi(-2) - initial.powi(2)) / a2;
    let h = (H / a2 - B * (x * x + y * y) / a2.powi(2)).max(0.0);
    CellState {
        h,
        qx: h * strain * x,
        qy: h * strain * y,
    }
}

#[test]
fn radial_moving_shoreline_error_decreases_under_refinement()
-> Result<(), Box<dyn std::error::Error>> {
    // Continuity gives h=H/a²-b*r²/a⁴. Momentum reduces to a''+omega²*a=omega²/a³.
    // Relative depth L1 stays below 0.1, with mean component discharge error below 0.005 m²/s.
    let mut previous = f64::INFINITY;
    for n in [40, 80] {
        let dx = 4.0 / n as f64;
        let centres: Vec<_> = (0..n)
            .flat_map(|row| {
                (0..n).map(move |column| {
                    (
                        -2.0 + (column as f64 + 0.5) * dx,
                        -2.0 + (row as f64 + 0.5) * dx,
                    )
                })
            })
            .collect();
        let domain = Domain::new(
            CartesianGrid::new(n, n, dx, dx)?,
            centres.iter().map(|&(x, y)| B * (x * x + y * y)).collect(),
            vec![0.0; n * n],
            vec![true; n * n],
            SpatialReference::new(
                "synthetic projected metres".into(),
                "synthetic SI datum".into(),
                [-2.0, -2.0],
            )?,
        )?;
        let state = FlowState::new(
            &domain,
            centres.iter().map(|&(x, y)| exact(x, y, 0.0)).collect(),
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 1.0,
                rain: vec![0.0; n * n],
                withdrawal: vec![0.0; n * n],
                boundaries: vec![],
            }],
        )?;
        let mut solver = Solver::new(
            domain,
            state,
            boundaries,
            forcing,
            SolverConfig::default(),
            0.0,
        )?;
        assert_eq!(
            solver.advance_to(0.5, 100_000)?.status,
            AdvanceStatus::ReachedTarget
        );
        let ledger = solver.ledger();
        assert!(ledger.residual.abs() <= 1e-9 + 1e-12 * ledger.initial_storage.max(1.0));
        assert_eq!(ledger.water_adjustment, 0.0);
        let mut depth_error = 0.0;
        let mut exact_depth = 0.0;
        let mut discharge_error = 0.0;
        for (cell, &(x, y)) in solver.state().cells().iter().zip(&centres) {
            let expected = exact(x, y, 0.5);
            assert!(cell.h >= 0.0);
            depth_error += (cell.h - expected.h).abs();
            exact_depth += expected.h;
            discharge_error += (cell.qx - expected.qx).abs() + (cell.qy - expected.qy).abs();
        }
        let relative = depth_error / exact_depth;
        let discharge_mean = discharge_error / (2 * n * n) as f64;
        println!(
            "Radial: n={n}, relative_depth_l1={relative:.12e}, mean_component_q_error={discharge_mean:.12e}"
        );
        assert!(
            relative < 0.1 && relative < previous,
            "n={n}, relative={relative}, previous={previous}"
        );
        assert!(
            discharge_mean < 0.005,
            "n={n}, discharge_mean={discharge_mean}"
        );
        previous = relative;
    }
    Ok(())
}
