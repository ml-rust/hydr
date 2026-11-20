//! Surveyed channel/pool oscillation against the linear-mode reference.
//!
//! The rectangle profile at 50, 100, and 200 cells with `CFL 0.1`, `g = h0 = b = 1`,
//! `L = A_pool = 10`, closed left end, `QuiescentReservoir` right end, `k = 0`. Frozen
//! gates from `linear-mode-reference.json`: `finest_limit 0.05`, `refinement_factor 0.8`,
//! `normalized_floor 1e-7`.

use super::fixture::{Result, batch, check_water, forcing};
use super::reference::CASES;
use super::surveyed_fixture::{definition, single, surveyed_case};
use super::surveyed_profiles::RECTANGLE_1;
use hydr::channel::{ChannelBoundary, ChannelInitialState};
use hydr::coupling::ChannelPoolEndpoint;

const AMPLITUDE: f64 = 1e-5;
const POOL_PEAK: f64 = 4.421205929549984e-6;
const K: f64 = 0.20287578381104342;
const QUARTER: f64 = 7.742650686480755;
const HALF: f64 = 15.48530137296151;
const FLOOR: f64 = 1e-7;
const FINEST_LIMIT: f64 = 0.05;
const REFINEMENT_FACTOR: f64 = 0.8;

fn average_cos(left: f64, right: f64) -> f64 {
    ((K * right).sin() - (K * left).sin()) / (K * (right - left))
}

fn average_sin(left: f64, right: f64) -> f64 {
    ((K * left).cos() - (K * right).cos()) / (K * (right - left))
}

/// Normalized RMS depth, RMS discharge, and pool stage errors at the quarter and half period.
fn measure(count: usize) -> Result<[[f64; 3]; 2]> {
    let mut case = surveyed_case(&CASES[0]);
    case.profile = RECTANGLE_1;
    case.gravity = 1.0;
    case.interior_area_m2 = 1.0;
    case.elevation = 1.0 + AMPLITUDE * (10.0 * K).cos();
    case.pool_area = 10.0;
    case.loss = 0.0;
    let (mut setup, mut initial) = single(&case, ChannelPoolEndpoint::Right)?;
    let dx = 10.0 / count as f64;
    setup.reaches[0].definition = definition(RECTANGLE_1, 1.0, count, dx);
    setup.reaches[0].definition.config.cfl = 0.1;
    initial.reaches[0] = ChannelInitialState {
        time_s: 0.0,
        volume_m3: (0..count)
            .map(|i| dx * (1.0 + AMPLITUDE * average_cos(i as f64 * dx, (i + 1) as f64 * dx)))
            .collect(),
        longitudinal_momentum_m4_per_s: vec![0.0; count],
    };
    let mut current = batch(setup.clone(), initial)?;
    let mut errors = [[0.0; 3]; 2];
    for (sample, end) in [QUARTER, HALF].into_iter().enumerate() {
        let mut steps = 0;
        while current.time() < end {
            let start = current.time();
            let mut held = forcing(&setup, start, end);
            held.reaches[0].left = Some(ChannelBoundary::Wall);
            current = current.trial_to(end, &held)?.into_next();
            assert!(current.time() > start && current.time() <= end);
            check_water(&current)?;
            steps += 1;
            assert!(steps < 1_000_000, "Surveyed finite-pool mode stalls");
        }
        assert_eq!(current.time().to_bits(), end.to_bits());
        let state = current.reach("reach").ok_or("Missing reach")?.state()?;
        for (i, cell) in state.iter().enumerate() {
            let left = i as f64 * dx;
            let right = (i + 1) as f64 * dx;
            let depth = 1.0 + AMPLITUDE * average_cos(left, right) * (K * end).cos();
            let discharge = AMPLITUDE * average_sin(left, right) * (K * end).sin();
            errors[sample][0] += (cell.depth_m - depth).powi(2) / count as f64;
            errors[sample][1] += (cell.discharge_m3_per_s - discharge).powi(2) / count as f64;
        }
        errors[sample][0] = errors[sample][0].sqrt() / AMPLITUDE;
        errors[sample][1] = errors[sample][1].sqrt() / AMPLITUDE;
        let pool_stage = current.pools().outputs()?[0]
            .stage_m
            .ok_or("Linear mode pool dries")?;
        let exact_pool = 1.0 + AMPLITUDE * (10.0 * K).cos() * (K * end).cos();
        errors[sample][2] = (pool_stage - exact_pool).abs() / POOL_PEAK;
    }
    Ok(errors)
}

#[test]
fn surveyed_linear_channel_pool_mode_refines_against_fixed_peak_amplitudes() -> Result {
    let errors = [measure(50)?, measure(100)?, measure(200)?];
    for sample in 0..2 {
        for quantity in 0..3 {
            for pair in errors.windows(2) {
                assert!(
                    pair[1][sample][quantity] <= REFINEMENT_FACTOR * pair[0][sample][quantity]
                        || pair[1][sample][quantity] <= FLOOR,
                    "Mode refinement sample={sample}, quantity={quantity}: {errors:?}"
                );
            }
            assert!(
                errors[2][sample][quantity] < FINEST_LIMIT,
                "Frozen mode ceiling sample={sample}, quantity={quantity}: {errors:?}"
            );
        }
    }
    Ok(())
}
