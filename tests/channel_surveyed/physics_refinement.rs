//! Temporal and combined refinement of surveyed transport against the frozen references.

use super::fixtures::Case;
use super::physics::{
    FINEST_LIMIT, Measurement, decreasing, differences, front_convergence, run, spatial_case,
};
use super::support::TestResult;

/// Fixed spacing for the temporal gate, which compares successive runs.
const TEMPORAL_CELLS: usize = 10;
/// Halving steps for the temporal gate, each under the CFL limit of the temporal mesh. The
/// coarsest run takes two steps, so its projection error is not zero.
const TEMPORAL_STEPS_S: [f64; 3] = [1.0, 0.5, 0.25];
/// Doubling meshes and halving steps for the combined gate. Each step stays under the CFL
/// limit of its mesh. The coarsest run takes four steps.
const COMBINED_RUNS: [(usize, f64); 3] = [(25, 0.5), (50, 0.25), (100, 0.125)];
/// Combined runs for a dry front. The first-order support edge spreads one cell per step, so
/// each run spreads `2 s * dx / dt = 50 m`. That passes the exact front, so the dry-side
/// volume of the coarsest run is not zero.
const DRY_FRONT_RUNS: [(usize, f64); 3] = [(20, 0.4), (40, 0.2), (80, 0.1)];

/// Runs `case` once per `(cells, max_step)` pair.
fn runs(
    case: Case,
    settings: impl IntoIterator<Item = (usize, f64)>,
) -> TestResult<Vec<Measurement>> {
    settings
        .into_iter()
        .map(|(count, max_step)| {
            run::<crate::runtime::Runtime>(crate::runtime::client(), case, count, max_step)
        })
        .collect()
}

/// Requires the conserved differences between successive halved steps to strictly decrease.
fn temporal(id: &str) -> TestResult {
    let case = spatial_case(id)?;
    let measurements = runs(case, TEMPORAL_STEPS_S.map(|step| (TEMPORAL_CELLS, step)))?;
    let deltas: Vec<_> = measurements
        .windows(2)
        .map(|pair| differences(case, &pair[0], &pair[1]))
        .collect();
    println!("surveyed {id} temporal differences: {deltas:?}");
    for component in 0..2 {
        decreasing(
            id,
            &deltas.iter().map(|row| row[component]).collect::<Vec<_>>(),
        );
    }
    Ok(())
}

/// Requires errors, restricted differences, and front distances to strictly decrease.
fn combined(id: &str, settings: [(usize, f64); 3]) -> TestResult {
    let case = spatial_case(id)?;
    let measurements = runs(case, settings)?;
    let deltas: Vec<_> = measurements
        .windows(2)
        .map(|pair| differences(case, &pair[0], &pair[1]))
        .collect();
    for component in 0..2 {
        decreasing(
            id,
            &measurements
                .iter()
                .map(|row| row.errors[component])
                .collect::<Vec<_>>(),
        );
        decreasing(
            id,
            &deltas.iter().map(|row| row[component]).collect::<Vec<_>>(),
        );
        assert!(measurements[2].errors[component] < FINEST_LIMIT);
    }
    front_convergence(case, &measurements)
}

#[test]
fn trapezoid_wet_fixed_spacing_temporal_conserved_differences_strictly_decrease() -> TestResult {
    temporal("trapezoid_wet")
}

#[test]
fn multiband_dry_fan_fixed_spacing_temporal_conserved_differences_strictly_decrease() -> TestResult
{
    temporal("both_knots_dry_fan")
}

#[test]
fn trapezoid_wet_combined_refinement_reduces_errors_and_restricted_conserved_differences()
-> TestResult {
    combined("trapezoid_wet", COMBINED_RUNS)
}

#[test]
fn multiband_dry_fan_combined_refinement_reduces_errors_and_restricted_conserved_differences()
-> TestResult {
    combined("both_knots_dry_fan", DRY_FRONT_RUNS)
}
