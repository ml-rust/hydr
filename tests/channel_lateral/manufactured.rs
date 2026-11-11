use super::fixtures::{binding, inflow, rate};
use crate::common::{TestResult, cell, check_water, definition, exterior, reach};
use hydr::channel::{ChannelForcingInterval, ChannelLateralDirection};

struct Solution {
    depths: Vec<f64>,
    discharges: Vec<f64>,
    depth_error: f64,
    discharge_error: f64,
}

fn solve(nx: usize, max_step: f64) -> Result<Solution, Box<dyn std::error::Error>> {
    let dx = 100.0 / nx as f64;
    let centres: Vec<_> = (0..nx).map(|i| (i as f64 + 0.5) * dx).collect();
    let mut layout = definition((0..nx).map(|i| cell(i, dx, 2.0, 0.0)).collect(), max_step);
    layout.lateral_bindings = (0..nx)
        .map(|i| {
            binding(
                &format!("input-{i}"),
                &format!("cell-{i}"),
                ChannelLateralDirection::Inflow,
            )
        })
        .collect();
    let mut result = reach(layout, &vec![0.0; nx], &vec![0.0; nx])?;
    let target = 0.1;
    let mut count = 0usize;
    while result.time() < target {
        let start = result.time();
        let lateral_inputs = centres
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let velocity = (20.0 + 0.2 * x) * (1.0 + 0.4 * start) / (1.0 + 0.2 * start);
                inflow(
                    &format!("input-{i}"),
                    rate(2.0 * (1.0 + 0.2 * start) * dx),
                    velocity,
                )
            })
            .collect();
        let interval = ChannelForcingInterval {
            start_s: start,
            end_s: target,
            left: exterior(start, if start == 0.0 { 0.0 } else { 20.0 }),
            right: exterior(start, if start == 0.0 { 0.0 } else { 40.0 }),
            lateral_inputs,
        };
        let next = result.step(target, &interval)?.next;
        assert!(next.time() > start && next.time() <= target);
        check_water(&next)?;
        result = next;
        count += 1;
        assert!(count < 1_000_000);
    }
    assert_eq!(result.time().to_bits(), target.to_bits());
    let state = result.state()?;
    let depth_error = state
        .iter()
        .map(|cell| (cell.depth_m - target).abs())
        .sum::<f64>()
        / nx as f64;
    let discharge_error = state
        .iter()
        .zip(&centres)
        .map(|(cell, &x)| (cell.discharge_m3_per_s - 2.0 * target * (20.0 + 0.2 * x)).abs())
        .sum::<f64>()
        / nx as f64;
    println!(
        "channel distributed inflow: nx={nx}, max_step={max_step}, t={target}, mean_abs_h={depth_error:.12e}, mean_abs_Q={discharge_error:.12e}"
    );
    assert!(depth_error < 0.01);
    assert!(discharge_error < 0.1);
    Ok(Solution {
        depths: state.iter().map(|cell| cell.depth_m).collect(),
        discharges: state.iter().map(|cell| cell.discharge_m3_per_s).collect(),
        depth_error,
        discharge_error,
    })
}

#[test]
fn distributed_inflow_depth_and_discharge_errors_decrease_with_mesh_spacing() -> TestResult {
    let mut results = Vec::new();
    for nx in [50, 100, 200] {
        results.push(solve(nx, 0.0005)?);
    }
    assert!(
        results
            .windows(2)
            .all(|pair| pair[1].depth_error < pair[0].depth_error)
    );
    assert!(
        results
            .windows(2)
            .all(|pair| pair[1].discharge_error < pair[0].discharge_error)
    );
    Ok(())
}

#[test]
fn distributed_inflow_normalized_differences_decrease_with_timestep() -> TestResult {
    let mut results = Vec::new();
    for max_step in [0.002, 0.001, 0.0005] {
        results.push(solve(200, max_step)?);
    }
    let differences: Vec<_> = results
        .windows(2)
        .map(|pair| {
            let h = pair[0]
                .depths
                .iter()
                .zip(&pair[1].depths)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / 200.0;
            let q = pair[0]
                .discharges
                .iter()
                .zip(&pair[1].discharges)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                / 200.0;
            h / 0.1 + q / 8.0
        })
        .collect();
    println!(
        "channel distributed inflow temporal: successive_normalized_differences={differences:?}"
    );
    assert!(differences[1] < differences[0]);
    Ok(())
}
