//! Certified surveyed scalar endpoint traces against the frozen 150-digit references.
//!
//! Frozen criteria:
//!
//! - Trace identity, rectangle and triangle: the frozen `channel_reservoir` fixtures and the
//!   `channel_boundaries` literal rarefaction and shock traces, rerun on the rectangle and
//!   triangle profiles. Each reference lies inside the certified depth enclosure. Nominal depth,
//!   velocity, discharge, and momentum flux within `1e-11` absolute plus `1e-12` relative.
//! - Multi-band traces: three bands and wide bottom. `Stage`, `Discharge`, and
//!   `QuiescentReservoir` (`k = 0`, `0.5`), entering and leaving, both ends. Trace depths in
//!   every band, and one within `1e-6 m` of each knot. Reference inside the enclosure. Nominal
//!   within `1e-11` plus `1e-12` relative. Flux widths within the configured budgets. The
//!   rows run are the first rows that cover each policy with each end and direction, each
//!   policy in each band, and each knot, plus the row with the largest nominal error.
//!
//! The certified depth enclosure is public for reservoir rows only, through
//! `ChannelReservoirTrace`. A `Stage` enclosure is the two-sum enclosure of `η - z_min`. Every
//! reference profile has `z_min = 0`, so that enclosure is the exact elevation. The in-crate
//! trace tests check the `Discharge` root enclosures against the same references.

use super::endpoints_reference::{
    End, IDENTITY_LITERAL, IDENTITY_RESERVOIR, MULTIBAND, Policy, Trace, Value,
};
use super::support::{TestResult, forcing, profile, reach};
use hydr::channel::{
    ChannelBoundary, ChannelBoundaryTrace, ChannelConfig, ChannelRiemannDiagnostics,
    ChannelSurveyedConfig,
};

/// Frozen nominal allowance: `1e-11` absolute plus `1e-12` relative.
fn allowance(reference: f64) -> f64 {
    1e-11 + 1e-12 * reference.abs()
}

/// One cell length. The face trace depends on the end-cell state only.
const LENGTH: f64 = 1.0;

/// A duration far below the end-face CFL limit, so the interval takes one step.
const DURATION: f64 = 1.0 / 1048576.0;

/// Largest nominal error of one row, relative to its allowance.
struct Observed {
    worst: f64,
}

fn close(label: &str, row: usize, actual: f64, reference: Value, observed: &mut Observed) {
    let error = (actual - reference.nearest).abs();
    let limit = allowance(reference.nearest);
    observed.worst = observed.worst.max(error);
    assert!(
        error <= limit,
        "row {row} {label}: actual={actual:e}, reference={:e}, error={error:e}, allowance={limit:e}",
        reference.nearest
    );
}

/// The receipt trace and Riemann diagnostics of `row` after one short interval.
fn solve(
    row: &Trace,
) -> TestResult<(
    ChannelBoundaryTrace,
    ChannelRiemannDiagnostics,
    ChannelConfig,
)> {
    let config = ChannelConfig {
        gravity: row.gravity,
        ..ChannelConfig::default()
    };
    let section = profile(row.profile.name, row.profile.points, row.profile.cap);
    let origin = reach::<crate::runtime::Runtime>(
        crate::runtime::client(),
        section,
        &[LENGTH],
        &[0.0],
        &[row.interior_area_m2],
        &[row.interior_discharge_m3_per_s],
        config.clone(),
        ChannelSurveyedConfig::default(),
    )?;
    let policy = row.policy.boundary();
    let (side, boundaries) = match row.end {
        End::Left => (0, [policy, ChannelBoundary::Wall]),
        End::Right => (1, [ChannelBoundary::Wall, policy]),
    };
    let next = origin
        .step(DURATION, &forcing(0.0, DURATION, boundaries))?
        .next;
    assert_eq!(next.time().to_bits(), DURATION.to_bits(), "one interval");
    let receipt = next.last_step()?.ok_or("missing endpoint receipt")?;
    let trace = receipt.boundary_traces[side]
        .clone()
        .ok_or("missing surveyed scalar trace")?;
    assert!(receipt.boundary_traces[1 - side].is_none());
    let riemann = receipt
        .riemann
        .clone()
        .ok_or("missing surveyed Riemann receipt")?;
    Ok((trace, riemann, config))
}

/// Checks one row and returns its largest nominal error.
fn check(index: usize, row: &Trace) -> TestResult<f64> {
    let (trace, riemann, config) = solve(row)?;
    let mut observed = Observed { worst: 0.0 };
    close("depth", index, trace.depth_m, row.depth, &mut observed);
    close(
        "velocity",
        index,
        trace.outward_velocity_m_per_s,
        row.outward_velocity,
        &mut observed,
    );
    close(
        "discharge",
        index,
        trace.outward_discharge_m3_per_s,
        row.outward_discharge,
        &mut observed,
    );
    close(
        "momentum flux",
        index,
        trace.momentum_flux_m4_per_s2,
        row.momentum_flux,
        &mut observed,
    );
    close("area", index, trace.area_m2, row.area, &mut observed);
    assert_eq!(
        trace.outward_discharge_m3_per_s > 0.0,
        row.leaving,
        "row {index} direction"
    );
    assert!(!trace.cap_compatibility_approximation);
    match row.policy {
        Policy::Stage { elevation_m } => {
            // z_min = 0: the two-sum enclosure is the exact elevation.
            assert_eq!(
                trace.depth_m.to_bits(),
                elevation_m.to_bits(),
                "row {index}"
            );
            assert!(row.depth.floor() == elevation_m && row.depth.ceil() == elevation_m);
            assert!(trace.reservoir.is_none());
        }
        Policy::Discharge { discharge_m3_per_s } => {
            let outward = if row.end == End::Left {
                -discharge_m3_per_s
            } else {
                discharge_m3_per_s
            };
            assert_eq!(
                trace.outward_discharge_m3_per_s.to_bits(),
                outward.to_bits()
            );
            assert_eq!(trace.water_flux_width_m3_per_s.to_bits(), 0.0_f64.to_bits());
            assert!(trace.reservoir.is_none());
        }
        Policy::Reservoir { .. } => {
            let reservoir = trace
                .reservoir
                .as_ref()
                .ok_or("missing reservoir receipt")?;
            assert!(
                reservoir.depth_lower_m <= row.depth.floor()
                    && row.depth.ceil() <= reservoir.depth_upper_m,
                "row {index}: enclosure [{:e}, {:e}] excludes the reference [{:e}, {:e}]",
                reservoir.depth_lower_m,
                reservoir.depth_upper_m,
                row.depth.floor(),
                row.depth.ceil()
            );
        }
    }
    // Flux widths within the configured budgets.
    let face = if row.end == End::Left { 0 } else { 1 };
    let water_width = riemann.water_flux_width_m3_per_s[face];
    let momentum_width = riemann.momentum_flux_width_m4_per_s2[face];
    assert_eq!(
        water_width.to_bits(),
        trace.water_flux_width_m3_per_s.to_bits()
    );
    assert_eq!(
        momentum_width.to_bits(),
        trace.momentum_flux_width_m4_per_s2.to_bits()
    );
    assert!(
        water_width
            <= config.riemann_water_flux_absolute_m3_per_s
                + config.riemann_flux_relative * trace.outward_discharge_m3_per_s.abs()
    );
    assert!(
        momentum_width
            <= config.riemann_momentum_flux_absolute_m4_per_s2
                + config.riemann_flux_relative * trace.momentum_flux_m4_per_s2.abs()
    );
    let work = riemann.surveyed_work.ok_or("missing surveyed work")?;
    assert!(work.maximum_simpson_panels >= ChannelSurveyedConfig::default().initial_simpson_panels);
    if matches!(row.policy, Policy::Stage { .. }) {
        assert_eq!(trace.root_iterations, 0);
    }
    Ok(observed.worst)
}

/// Checks every `(index, row)` and prints the largest nominal error and its row.
fn run<'a>(rows: impl Iterator<Item = (usize, &'a Trace)>, label: &str) -> TestResult {
    let (mut worst, mut worst_row): (f64, usize) = (0.0, 0);
    let mut count = 0;
    for (index, row) in rows {
        let error = check(index, row).map_err(|error| format!("{label} row {index}: {error}"))?;
        if error > worst {
            (worst, worst_row) = (error, index);
        }
        count += 1;
    }
    println!("{label}: rows={count}, worst nominal error={worst:e} at row {worst_row}");
    Ok(())
}

/// Policy kind: stage, discharge, reservoir `k = 0`, reservoir `k > 0`.
fn kind(policy: Policy) -> u8 {
    match policy {
        Policy::Stage { .. } => 0,
        Policy::Discharge { .. } => 1,
        Policy::Reservoir {
            entrance_loss_coefficient,
            ..
        } => 2 + u8::from(entrance_loss_coefficient > 0.0),
    }
}

/// `MULTIBAND` rows with the largest nominal error over each full profile: three bands, then
/// wide bottom.
const HARDEST: [usize; 2] = [141, 245];

/// The rows that first cover each policy kind with each end and direction, each policy kind
/// in each band, and each interior knot within `1e-6 m`, plus the `HARDEST` rows.
fn covering<'a>(
    rows: impl Iterator<Item = (usize, &'a Trace)>,
    knots: &[f64],
) -> Vec<(usize, &'a Trace)> {
    let mut directions = std::collections::BTreeSet::new();
    let mut bands = std::collections::BTreeSet::new();
    let mut near = vec![false; knots.len()];
    rows.filter(|&(index, row)| {
        let policy = kind(row.policy);
        let direction = directions.insert((policy, row.end == End::Left, row.leaving));
        let band = bands.insert((policy, row.band));
        let mut knot = false;
        for (seen, &value) in near.iter_mut().zip(knots) {
            if !*seen && (row.depth.nearest - value).abs() <= 1e-6 {
                *seen = true;
                knot = true;
            }
        }
        direction || band || knot || HARDEST.contains(&index)
    })
    .collect()
}

/// The covering multi-band rows of profile `name`: trace depths in every band, one within
/// `1e-6 m` of each knot, then every selected row against its reference.
fn multiband(name: &str) -> TestResult {
    let all = || {
        MULTIBAND
            .iter()
            .enumerate()
            .filter(move |(_, row)| row.profile.name == name)
    };
    let (_, first) = all().next().ok_or("missing profile rows")?;
    let knots = first.profile.knots;
    let selected = covering(all(), &knots[1..knots.len() - 1]);
    let rows = || selected.iter().copied();
    let directions: std::collections::BTreeSet<_> = rows()
        .map(|(_, row)| (kind(row.policy), row.end == End::Left, row.leaving))
        .collect();
    assert_eq!(
        directions.len(),
        16,
        "{name}: a policy, end, and direction holds no trace"
    );
    let bands: std::collections::BTreeSet<_> = rows().map(|(_, row)| row.band).collect();
    assert_eq!(
        bands.len(),
        knots.len() - 1,
        "{name}: a band holds no trace"
    );
    for &knot in &knots[1..knots.len() - 1] {
        assert!(
            rows().any(|(_, row)| (row.depth.nearest - knot).abs() <= 1e-6),
            "{name}: no trace within 1e-6 m of knot {knot}"
        );
    }
    run(rows(), name)
}

#[test]
fn rectangle_and_triangle_reservoir_traces_match_the_frozen_fixtures() -> TestResult {
    run(IDENTITY_RESERVOIR.iter().enumerate(), "identity reservoir")
}

#[test]
fn rectangle_and_triangle_literal_traces_match_the_frozen_fixtures() -> TestResult {
    run(IDENTITY_LITERAL.iter().enumerate(), "identity literal")
}

#[test]
fn three_band_traces_match_the_frozen_references() -> TestResult {
    multiband("three-bands")
}

#[test]
fn wide_bottom_traces_match_the_frozen_references() -> TestResult {
    multiband("wide-bottom")
}
