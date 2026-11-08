//! Frozen analytical accuracy, refinement, and moving-shoreline acceptance cases.
//!
//! Vacuum uses exact supercritical far-field flux ports because its initial states flow through both channel ends.

mod support;

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    AdvanceStatus, Axis, BoundaryConditions, BoundaryFace, FluxPort, ForcingInterval,
    ForcingSchedule, Solver, SolverConfig,
};
use support::analytic::{ritter, stoker, thacker, vacuum};

type TestResult = Result<(), Box<dyn std::error::Error>>;
type Reference = fn(f64, f64, f64) -> (f64, f64);
const GRAVITY: f64 = 9.81;

struct ChannelResult {
    error: f64,
    depth: Vec<f64>,
}

fn reference(origin: [f64; 2]) -> Result<SpatialReference, Box<dyn std::error::Error>> {
    Ok(SpatialReference::new(
        "synthetic projected metres".into(),
        "synthetic SI datum".into(),
        origin,
    )?)
}

fn advance(
    domain: Domain,
    cells: Vec<CellState>,
    ports: Vec<FluxPort>,
    target: f64,
    max_step: f64,
) -> Result<Solver, Box<dyn std::error::Error>> {
    let count = domain.grid().cell_count();
    let boundaries = if ports.is_empty() {
        BoundaryConditions::impermeable_walls()
    } else {
        BoundaryConditions::with_ports(&domain, ports.iter().map(|port| port.face).collect())?
    };
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: target + 1.0,
            rain: vec![0.0; count],
            withdrawal: vec![0.0; count],
            ports,
        }],
    )?;
    let state = FlowState::new(&domain, cells)?;
    let mut solver = Solver::new(
        domain,
        state,
        boundaries,
        forcing,
        SolverConfig {
            gravity: GRAVITY,
            max_step,
            ..SolverConfig::default()
        },
        0.0,
    )?;
    assert_eq!(
        solver.advance_to(target, 1_000_000)?.status,
        AdvanceStatus::ReachedTarget
    );
    assert!(solver.state().cells().iter().all(|cell| cell.h >= 0.0));
    Ok(solver)
}

fn channel_error(
    label: &str,
    nx: usize,
    target: f64,
    max_step: f64,
    exact: Reference,
    ports: Vec<FluxPort>,
) -> Result<ChannelResult, Box<dyn std::error::Error>> {
    let dx = 200.0 / nx as f64;
    let domain = Domain::new(
        CartesianGrid::new(nx, 1, dx, 200.0)?,
        vec![0.0; nx],
        vec![0.0; nx],
        vec![true; nx],
        reference([-100.0, -100.0])?,
    )?;
    let centres: Vec<_> = (0..nx)
        .map(|column| -100.0 + (column as f64 + 0.5) * dx)
        .collect();
    let cells = centres
        .iter()
        .map(|&x| {
            let (h, qx) = exact(x, 0.0, GRAVITY);
            CellState { h, qx, qy: 0.0 }
        })
        .collect();
    // Dam fixtures have stationary remote walls. Separating fans supply exact far-field ports.
    let solver = advance(domain, cells, ports, target, max_step)?;
    let (depth_sum, discharge_sum) = solver.state().cells().iter().zip(&centres).fold(
        (0.0, 0.0),
        |(depth, discharge), (cell, &x)| {
            let (h, q) = exact(x, target, GRAVITY);
            (depth + (cell.h - h).abs(), discharge + (cell.qx - q).abs())
        },
    );
    let depth_error = depth_sum / nx as f64;
    println!(
        "{label}: nx={nx}, max_step={max_step}, t={target}, mean_abs_h={depth_error:.12e}, mean_abs_q={:.12e}",
        discharge_sum / nx as f64
    );
    Ok(ChannelResult {
        error: depth_error,
        depth: solver.state().cells().iter().map(|cell| cell.h).collect(),
    })
}

fn decreasing(label: &str, errors: &[f64]) {
    for pair in errors.windows(2) {
        assert!(
            pair[1] < pair[0],
            "{label}: refinement must decrease error, received {errors:?}"
        );
    }
}

#[test]
fn ritter_depth_error_decreases_under_spatial_refinement() -> TestResult {
    let mut errors = Vec::new();
    for nx in [100, 200, 400] {
        errors.push(channel_error("Ritter spatial", nx, 4.0, 1.0, ritter, vec![])?.error);
    }
    assert!(errors.iter().all(|&error| error < 0.06), "{errors:?}");
    decreasing("Ritter spatial", &errors);
    Ok(())
}

#[test]
fn stoker_depth_error_decreases_under_spatial_refinement() -> TestResult {
    let mut errors = Vec::new();
    for nx in [100, 200, 400] {
        errors.push(channel_error("Stoker spatial", nx, 4.0, 1.0, stoker, vec![])?.error);
    }
    assert!(errors.iter().all(|&error| error < 0.06), "{errors:?}");
    decreasing("Stoker spatial", &errors);
    Ok(())
}

#[test]
fn separating_fans_preserve_nonnegative_depth_and_match_vacuum() -> TestResult {
    let width = 200.0;
    let ports = [(0, -1.0), (400, 1.0)]
        .into_iter()
        .map(|(column, outward)| FluxPort {
            face: BoundaryFace {
                axis: Axis::X,
                column,
                row: 0,
            },
            water_rate: 8.0 * width,
            momentum_rate: [outward * (64.0 + 0.5 * GRAVITY) * width, 0.0],
            signal_speed: 12.0,
            provenance: "Exact supercritical far state h=1, u=-8 left/+8 right, qy=0; full advective/pressure flux; |u|+sqrt(gh)<12".into(),
        })
        .collect();
    let error = channel_error("Vacuum", 400, 1.0, 1.0, vacuum, ports)?.error;
    assert!(error < 0.06, "Vacuum mean absolute depth error {error}");
    Ok(())
}

#[test]
fn ritter_reports_separate_temporal_and_combined_refinement() -> TestResult {
    let mut temporal = Vec::new();
    for max_step in [0.02, 0.01, 0.005] {
        temporal.push(channel_error(
            "Ritter temporal",
            400,
            4.0,
            max_step,
            ritter,
            vec![],
        )?);
    }
    let differences: Vec<_> = temporal
        .windows(2)
        .map(|pair| {
            pair[0]
                .depth
                .iter()
                .zip(&pair[1].depth)
                .map(|(coarse, fine)| (coarse - fine).abs())
                .sum::<f64>()
                / pair[0].depth.len() as f64
        })
        .collect();
    println!("Ritter temporal: successive mean_abs_depth_differences={differences:?}");
    let mut combined = Vec::new();
    for (nx, max_step) in [(100, 0.02), (200, 0.01), (400, 0.005)] {
        combined.push(channel_error("Ritter combined", nx, 4.0, max_step, ritter, vec![])?.error);
    }
    assert!(
        temporal.iter().all(|result| result.error < 0.06),
        "Ritter temporal analytical depth errors exceed 0.06"
    );
    assert!(combined.iter().all(|&error| error < 0.06), "{combined:?}");
    decreasing("Ritter temporal Cauchy", &differences);
    decreasing("Ritter combined", &combined);
    Ok(())
}

#[test]
fn thacker_shoreline_depth_error_decreases_under_refinement() -> TestResult {
    let target = 0.5;
    let mut errors = Vec::new();
    for n in [40, 80] {
        let spacing = 4.0 / n as f64;
        let count = n * n;
        let centres: Vec<_> = (0..count)
            .map(|index| {
                (
                    -2.0 + (index % n) as f64 * spacing + 0.5 * spacing,
                    -2.0 + (index / n) as f64 * spacing + 0.5 * spacing,
                )
            })
            .collect();
        let bed = centres
            .iter()
            .map(|&(x, y)| 0.1 * (x * x + y * y))
            .collect();
        let domain = Domain::new(
            CartesianGrid::new(n, n, spacing, spacing)?,
            bed,
            vec![0.0; count],
            vec![true; count],
            reference([-2.0, -2.0])?,
        )?;
        let cells = centres
            .iter()
            .map(|&(x, y)| {
                let (h, qx, qy) = thacker(x, y, 0.0, GRAVITY);
                CellState { h, qx, qy }
            })
            .collect();
        let solver = advance(domain, cells, vec![], target, 1.0)?;
        let error = solver
            .state()
            .cells()
            .iter()
            .zip(&centres)
            .map(|(cell, &(x, y))| (cell.h - thacker(x, y, target, GRAVITY).0).abs())
            .sum::<f64>()
            / count as f64;
        println!("Thacker: n={n}, t={target}, mean_abs_h={error:.12e}");
        errors.push(error);
    }
    assert!(errors.iter().all(|&error| error < 0.01), "{errors:?}");
    decreasing("Thacker spatial", &errors);
    Ok(())
}

#[test]
fn manning_equilibrium_drift_decreases_under_refinement() -> TestResult {
    let mut depth_errors = Vec::new();
    let mut discharge_errors = Vec::new();
    for nx in [40, 80] {
        let dx = 100.0 / nx as f64;
        let width = 100.0;
        let domain = Domain::new(
            CartesianGrid::new(nx, 1, dx, width)?,
            (0..nx)
                .map(|column| -0.0009 * (column as f64 + 0.5) * dx)
                .collect(),
            vec![0.03; nx],
            vec![true; nx],
            reference([0.0, 0.0])?,
        )?;
        let ports = [(0, -1.0), (nx, 1.0)]
            .into_iter()
            .map(|(column, outward)| FluxPort {
                face: BoundaryFace {
                    axis: Axis::X,
                    column,
                    row: 0,
                },
                water_rate: outward * width,
                momentum_rate: [outward * (1.0 + 0.5 * GRAVITY) * width, 0.0],
                signal_speed: 4.2,
                provenance: "Manufactured uniform h=1, qx=1, qy=0, n=0.03, slope=0.0009; full advective/pressure flux; |u|+sqrt(gh)<4.2".into(),
            })
            .collect();
        let solver = advance(
            domain,
            vec![
                CellState {
                    h: 1.0,
                    qx: 1.0,
                    qy: 0.0
                };
                nx
            ],
            ports,
            5.0,
            1.0,
        )?;
        // The central half excludes both source-complete boundary stencils.
        let interior = &solver.state().cells()[nx / 4..3 * nx / 4];
        let depth_error = interior
            .iter()
            .map(|cell| (cell.h - 1.0).abs())
            .fold(0.0, f64::max);
        let discharge_error = interior
            .iter()
            .map(|cell| (cell.qx - 1.0).abs())
            .fold(0.0, f64::max);
        println!(
            "Manning: nx={nx}, t=5, max_interior_h_drift={depth_error:.12e}, max_interior_q_drift={discharge_error:.12e}"
        );
        depth_errors.push(depth_error);
        discharge_errors.push(discharge_error);
    }
    assert!(
        depth_errors.iter().all(|&error| error < 0.01),
        "{depth_errors:?}"
    );
    assert!(
        discharge_errors.iter().all(|&error| error < 0.01),
        "{discharge_errors:?}"
    );
    decreasing("Manning interior depth", &depth_errors);
    decreasing("Manning interior discharge", &discharge_errors);
    Ok(())
}
