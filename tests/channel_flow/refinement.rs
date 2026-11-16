use super::common::{GRAVITY, TestResult, advance, cell, definition, exterior, reach, walls};
use crate::analytic::{ritter, stoker};

type Reference = fn(f64, f64, f64) -> (f64, f64);

/// Dam comparison time, before either wave reaches a wall.
const DAM_TIME_S: f64 = 2.0;

struct DamResult {
    depth_error: f64,
    discharge_error: f64,
    depths: Vec<f64>,
}

fn dam(
    label: &str,
    nx: usize,
    max_step: f64,
    exact: Reference,
) -> Result<DamResult, Box<dyn std::error::Error>> {
    let dx = 200.0 / nx as f64;
    let centres: Vec<_> = (0..nx).map(|i| -100.0 + (i as f64 + 0.5) * dx).collect();
    let cells = (0..nx).map(|i| cell(i, dx, 2.0, 0.0)).collect();
    let depths: Vec<_> = centres.iter().map(|&x| exact(x, 0.0, GRAVITY).0).collect();
    let discharges: Vec<_> = centres
        .iter()
        .map(|&x| 2.0 * exact(x, 0.0, GRAVITY).1)
        .collect();
    let initial = reach(definition(cells, max_step), &depths, &discharges)?;
    let result = advance(initial, DAM_TIME_S, walls())?;
    let state = result.state()?;
    // Independent references return point values at cell centres, rather than exact cell averages.
    let depth_error = state
        .iter()
        .zip(&centres)
        .map(|(cell, &x)| (cell.depth_m - exact(x, DAM_TIME_S, GRAVITY).0).abs())
        .sum::<f64>()
        / nx as f64;
    let discharge_error = state
        .iter()
        .zip(&centres)
        .map(|(cell, &x)| (cell.discharge_m3_per_s - 2.0 * exact(x, DAM_TIME_S, GRAVITY).1).abs())
        .sum::<f64>()
        / nx as f64;
    println!(
        "channel {label}: nx={nx}, max_step={max_step}, t={DAM_TIME_S}, mean_abs_h={depth_error:.12e}, mean_abs_Q={discharge_error:.12e}"
    );
    Ok(DamResult {
        depth_error,
        discharge_error,
        depths: state.iter().map(|cell| cell.depth_m).collect(),
    })
}

fn decreasing(label: &str, errors: &[f64]) {
    assert!(
        errors.windows(2).all(|pair| pair[1] < pair[0]),
        "{label}: refinement errors={errors:?}"
    );
}

#[test]
fn wet_and_dry_dam_depth_errors_decrease_under_spatial_refinement() -> TestResult {
    for (label, exact) in [
        ("Ritter", ritter as Reference),
        ("Stoker", stoker as Reference),
    ] {
        let mut depth_errors = Vec::new();
        let mut discharge_errors = Vec::new();
        for nx in [100, 200, 400] {
            let result = dam(label, nx, 1.0, exact)?;
            depth_errors.push(result.depth_error);
            discharge_errors.push(result.discharge_error);
        }
        assert!(
            depth_errors.iter().all(|&error| error < 0.06),
            "{label}: {depth_errors:?}"
        );
        assert!(
            discharge_errors.iter().all(|&error| error < 0.3),
            "{label}: {discharge_errors:?}"
        );
        decreasing(label, &depth_errors);
        decreasing(label, &discharge_errors);
    }
    Ok(())
}

#[test]
fn dry_dam_reports_temporal_and_combined_refinement() -> TestResult {
    let mut temporal = Vec::new();
    for max_step in [0.02, 0.01, 0.005] {
        temporal.push(dam("Ritter temporal", 400, max_step, ritter)?);
    }
    let differences: Vec<_> = temporal
        .windows(2)
        .map(|pair| {
            pair[0]
                .depths
                .iter()
                .zip(&pair[1].depths)
                .map(|(coarse, fine)| (coarse - fine).abs())
                .sum::<f64>()
                / pair[0].depths.len() as f64
        })
        .collect();
    println!("channel Ritter temporal: successive mean_abs_depth_differences={differences:?}");
    assert!(temporal.iter().all(|result| result.depth_error < 0.06));
    assert!(temporal.iter().all(|result| result.discharge_error < 0.3));
    decreasing("Ritter temporal Cauchy", &differences);
    let mut combined_depth = Vec::new();
    let mut combined_discharge = Vec::new();
    for (nx, max_step) in [(100, 0.02), (200, 0.01), (400, 0.005)] {
        let result = dam("Ritter combined", nx, max_step, ritter)?;
        combined_depth.push(result.depth_error);
        combined_discharge.push(result.discharge_error);
    }
    assert!(
        combined_depth.iter().all(|&error| error < 0.06),
        "{combined_depth:?}"
    );
    assert!(
        combined_discharge.iter().all(|&error| error < 0.3),
        "{combined_discharge:?}"
    );
    decreasing("Ritter combined depth", &combined_depth);
    decreasing("Ritter combined discharge", &combined_discharge);
    Ok(())
}

#[test]
fn smooth_width_flow_drift_decreases_under_spatial_refinement() -> TestResult {
    let mut depth_errors = Vec::new();
    let mut discharge_errors = Vec::new();
    for nx in [40, 80, 160] {
        let dx = 100.0 / nx as f64;
        let widths: Vec<_> = (0..nx)
            .map(|i| {
                let x = (i as f64 + 0.5) * dx;
                2.0 + 0.2 * (std::f64::consts::TAU * x / 100.0).sin()
            })
            .collect();
        let cells = widths
            .iter()
            .enumerate()
            .map(|(i, &width)| cell(i, dx, width, -(1.0 / width).powi(2) / (2.0 * GRAVITY)))
            .collect();
        let initial = reach(definition(cells, 1.0), &vec![1.0; nx], &vec![1.0; nx])?;
        let result = advance(
            initial,
            5.0,
            [
                exterior(1.0, 1.0 / widths[0]),
                exterior(1.0, 1.0 / widths[nx - 1]),
            ],
        )?;
        let state = result.state()?;
        let depth_error = state
            .iter()
            .map(|cell| (cell.depth_m - 1.0).abs() * dx)
            .sum::<f64>()
            / 100.0;
        let discharge_error = state
            .iter()
            .map(|cell| (cell.discharge_m3_per_s - 1.0).abs() * dx)
            .sum::<f64>()
            / 100.0;
        println!(
            "channel smooth width: nx={nx}, t=5, mean_abs_h={depth_error:.12e}, mean_abs_Q={discharge_error:.12e}"
        );
        depth_errors.push(depth_error);
        discharge_errors.push(discharge_error);
    }
    assert!(
        depth_errors.iter().all(|&error| error < 0.01),
        "{depth_errors:?}"
    );
    assert!(
        discharge_errors.iter().all(|&error| error < 0.02),
        "{discharge_errors:?}"
    );
    decreasing("Smooth width depth", &depth_errors);
    decreasing("Smooth width discharge", &discharge_errors);
    Ok(())
}
