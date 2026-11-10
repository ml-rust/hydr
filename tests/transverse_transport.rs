//! Exact transverse shear advection and Cartesian rotation covariance.
//!
//! Fixture: flat bed, h=1, normal velocity=1, no friction or sources, gravity=9.81.
//! Tangential discharge is 0.1 exp(-((s + 20 - t)/10)^2), truncated to zero beyond five Gaussian widths.
//! This translated profile solves tangential transport exactly while depth and normal discharge remain one.
//! The domain is [-100,100]², with exact complete far-field ports at the normal ends.
//! Transverse boundaries are physical walls. Their disturbances are excluded from the shear error sample.
//! Shear grids contain 100/200/400 normal cells and 67 transverse cells, with 32 steps of 1/32 s.
//! The centre row lies 33 faces from either wall, beyond the 32-step finite-volume stencil dependency.
//! Acceptance requires strictly decreasing centre-row tangential MAE, each <=0.03 m²/s.
//! Relative L1, sum(abs(qy - exact)) / sum(abs(exact)), must remain <0.2 on each grid.
//! Centre-row depth and normal-discharge drift must each remain <=1e-10 in SI units.
//! A separate 67² square-grid test rotates the complete problem through 90 degrees, including boundary ports.
//! The maximum rotated state-component discrepancy must remain <=1e-11.
//! Printed peaks describe measured amplitude only. These gates establish no general attenuation bound.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "support/impervious.rs"]
mod impervious;

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    AdvanceStatus, Axis, BoundaryConditions, BoundaryFace, FluxPort, ForcingInterval,
    ForcingSchedule, Solver, SolverConfig,
};
use impervious::impervious_solver;

type TestResult = Result<(), Box<dyn std::error::Error>>;
const GRAVITY: f64 = 9.81;
const STEP: f64 = 1.0 / 32.0;
const STEPS: usize = 32;
const LENGTH: f64 = 200.0;

#[derive(Clone, Copy)]
enum Orientation {
    X,
    RotatedY,
}

fn shear(coordinate: f64, time: f64) -> f64 {
    let distance = coordinate + 20.0 - time;
    if distance.abs() > 50.0 {
        0.0
    } else {
        0.1 * (-(distance / 10.0).powi(2)).exp()
    }
}

fn centre(index: usize, spacing: f64) -> f64 {
    -0.5 * LENGTH + (index as f64 + 0.5) * spacing
}

fn run(
    nx: usize,
    ny: usize,
    orientation: Orientation,
) -> Result<Solver<crate::runtime::Runtime>, Box<dyn std::error::Error>> {
    let dx = LENGTH / nx as f64;
    let dy = LENGTH / ny as f64;
    let count = nx * ny;
    let domain = Domain::new(
        CartesianGrid::new(nx, ny, dx, dy)?,
        vec![0.0; count],
        vec![0.0; count],
        vec![true; count],
        SpatialReference::new(
            "synthetic projected metres".into(),
            "synthetic SI datum".into(),
            [-100.0, -100.0],
        )?,
    )?;
    let cells = (0..count)
        .map(|index| match orientation {
            Orientation::X => CellState {
                h: 1.0,
                qx: 1.0,
                qy: shear(centre(index % nx, dx), 0.0),
            },
            Orientation::RotatedY => CellState {
                h: 1.0,
                qx: -shear(centre(index / nx, dy), 0.0),
                qy: 1.0,
            },
        })
        .collect();
    let (normal_count, transverse_count, width) = match orientation {
        Orientation::X => (nx, ny, dy),
        Orientation::RotatedY => (ny, nx, dx),
    };
    let mut ports = Vec::with_capacity(2 * transverse_count);
    for transverse in 0..transverse_count {
        for (normal, outward) in [(0, -1.0), (normal_count, 1.0)] {
            let momentum = outward * (1.0 + 0.5 * GRAVITY) * width;
            let (face, momentum_rate) = match orientation {
                Orientation::X => (
                    BoundaryFace {
                        axis: Axis::X,
                        column: normal,
                        row: transverse,
                    },
                    [momentum, 0.0],
                ),
                Orientation::RotatedY => (
                    BoundaryFace {
                        axis: Axis::Y,
                        column: transverse,
                        row: normal,
                    },
                    [0.0, momentum],
                ),
            };
            ports.push(FluxPort {
                face,
                water_rate: outward * width,
                momentum_rate,
                signal_speed: 4.2,
                provenance: "Exact flat far state h=1, normal velocity=1, tangential velocity=0; complete advective and pressure flux; |u_normal|+sqrt(gh)<4.2".into(),
            });
        }
    }
    let boundaries =
        BoundaryConditions::with_open_faces(&domain, ports.iter().map(|port| port.face).collect())?;
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 2.0,
            potential_evaporation: vec![0.0; count],
            rain: vec![0.0; count],
            withdrawal: vec![0.0; count],
            boundaries: ports.into_iter().map(Into::into).collect(),
        }],
    )?;
    let state = FlowState::new(&domain, cells)?;
    let mut solver = impervious_solver(
        domain,
        state,
        boundaries,
        forcing,
        SolverConfig {
            gravity: GRAVITY,
            max_step: STEP,
            ..SolverConfig::default()
        },
        0.0,
    )?;
    let report = solver.advance_to(1.0, STEPS)?;
    assert_eq!(report.status, AdvanceStatus::ReachedTarget);
    assert_eq!(report.accepted_steps, STEPS);
    Ok(solver)
}

#[test]
fn transverse_shear_error_decreases_under_spatial_refinement() -> TestResult {
    let ny = 67;
    let centre_row = ny / 2;
    assert!(centre_row + 1 > STEPS);
    assert!(ny - centre_row > STEPS);
    let mut errors = Vec::new();
    for nx in [100, 200, 400] {
        let solver = run(nx, ny, Orientation::X)?;
        let flow = solver.state()?;
        let cells = &flow.cells()[centre_row * nx..(centre_row + 1) * nx];
        let dx = LENGTH / nx as f64;
        let mut error = 0.0;
        let mut exact_norm = 0.0;
        let mut depth_drift: f64 = 0.0;
        let mut normal_drift: f64 = 0.0;
        let mut measured_peak: f64 = 0.0;
        let mut sampled_exact_peak: f64 = 0.0;
        for (column, cell) in cells.iter().enumerate() {
            let exact = shear(centre(column, dx), 1.0);
            error += (cell.qy - exact).abs();
            exact_norm += exact.abs();
            depth_drift = depth_drift.max((cell.h - 1.0).abs());
            normal_drift = normal_drift.max((cell.qx - 1.0).abs());
            measured_peak = measured_peak.max(cell.qy);
            sampled_exact_peak = sampled_exact_peak.max(exact);
        }
        let relative_l1 = error / exact_norm;
        error /= nx as f64;
        println!(
            "Transverse shear: nx={nx}, ny={ny}, steps={STEPS}, mean_abs_qy={error:.12e}, relative_l1={relative_l1:.12e}, max_h_drift={depth_drift:.12e}, max_qx_drift={normal_drift:.12e}, measured_peak={measured_peak:.12e}, sampled_exact_peak={sampled_exact_peak:.12e}"
        );
        assert!(error <= 0.03, "nx={nx}: transverse MAE={error}");
        assert!(relative_l1 < 0.2, "nx={nx}: relative L1={relative_l1}");
        assert!(depth_drift <= 1e-10, "nx={nx}: depth drift={depth_drift}");
        assert!(normal_drift <= 1e-10, "nx={nx}: qx drift={normal_drift}");
        errors.push(error);
    }
    for pair in errors.windows(2) {
        assert!(
            pair[1] < pair[0],
            "Transverse refinement errors: {errors:?}"
        );
    }
    Ok(())
}

#[test]
fn quarter_turn_rotates_state_and_complete_boundary_fluxes() -> TestResult {
    let n = 67;
    let original = run(n, n, Orientation::X)?;
    let rotated = run(n, n, Orientation::RotatedY)?;
    let mut discrepancy: f64 = 0.0;
    for row in 0..n {
        for column in 0..n {
            let before = original.state()?.cells()[row * n + column];
            let after = rotated.state()?.cells()[column * n + (n - 1 - row)];
            discrepancy = discrepancy
                .max((after.h - before.h).abs())
                .max((after.qx + before.qy).abs())
                .max((after.qy - before.qx).abs());
        }
    }
    println!("Transverse quarter turn: n={n}, max_state_component_error={discrepancy:.12e}");
    assert!(discrepancy <= 1e-11, "Rotated state error={discrepancy}");
    Ok(())
}
