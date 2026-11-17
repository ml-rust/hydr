use super::fixtures::{CASES, Case};
use super::support::{self, TestResult, advance, profile, reach};
use hydr::channel::{ChannelBoundary, ChannelConfig, ChannelSurveyedConfig};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Doubling meshes for the convergence gates, each dividing the frozen 400-cell reference.
const MESHES: [usize; 3] = [100, 200, 400];
/// Time of the frozen exact references.
const REFERENCE_TIME_S: f64 = 2.0;
/// Step shared by every spatial mesh.
const SPATIAL_STEP_S: f64 = 0.1;
/// Fixed spacing for the temporal gate, which compares successive runs.
const TEMPORAL_CELLS: usize = 10;
/// Mesh for the native advancement check on each runtime.
const NATIVE_CELLS: usize = 20;
/// Area fraction of the area scale below which a cell counts as dry.
const WET_FRACTION: f64 = 1e-12;

struct Measurement {
    errors: [f64; 2],
    conserved: Vec<[f64; 2]>,
}

fn boundaries(case: Case) -> [ChannelBoundary; 2] {
    case.exterior.map(|state| ChannelBoundary::Exterior {
        depth_m: f64::from_bits(state[0]),
        velocity_m_per_s: f64::from_bits(state[1]),
    })
}

fn run<R>(client: R::Client, case: Case, count: usize, max_step: f64) -> TestResult<Measurement>
where
    R: Runtime<DType = DType>,
    R::Client: support::Client<R>,
{
    println!(
        "surveyed {}: starting cells={count}, max_step={max_step}",
        case.id
    );
    let dx = 200.0 / count as f64;
    let initial: Vec<_> = (0..count)
        .map(|index| case.initial[usize::from(index >= count / 2)].map(f64::from_bits))
        .collect();
    let areas: Vec<_> = initial.iter().map(|state| state[0]).collect();
    let discharges: Vec<_> = initial.iter().map(|state| state[1]).collect();
    let current = reach::<R>(
        client,
        profile(case.id, case.points, case.cap),
        &vec![dx; count],
        &vec![0.0; count],
        &areas,
        &discharges,
        ChannelConfig {
            gravity: f64::from_bits(case.gravity),
            max_step,
            ..ChannelConfig::default()
        },
        ChannelSurveyedConfig::default(),
    )?;
    let next = advance(current, REFERENCE_TIME_S, boundaries(case))?;
    let conserved: Vec<_> = next
        .state()?
        .iter()
        .map(|cell| {
            [
                cell.volume_m3 / dx,
                cell.longitudinal_momentum_m4_per_s / dx,
            ]
        })
        .collect();
    let errors = errors(case, &conserved, &restricted_reference(case, count)?)?;
    let last = next
        .last_step()?
        .ok_or("Missing surveyed transport receipt")?;
    let riemann = last
        .riemann
        .as_ref()
        .ok_or("Missing surveyed Riemann diagnostics")?;
    assert!(riemann.surveyed_work.is_some());
    assert_eq!(last.left_pressure_impulse_m4_per_s, vec![0.0; count + 1]);
    assert_eq!(last.right_pressure_impulse_m4_per_s, vec![0.0; count + 1]);
    assert!(
        riemann.root_iterations
            <= ChannelSurveyedConfig::default()
                .normalized_root
                .max_iterations
    );
    println!(
        "surveyed {}: cells={count}, max_step={max_step}, normalized_E1={errors:?}",
        case.id
    );
    Ok(Measurement { errors, conserved })
}

/// Frozen 400-cell reference cells per cell of a `count`-cell mesh.
fn reference_ratio(case: Case, count: usize) -> TestResult<usize> {
    let fine = case.grids[2].len();
    if count == 0 || !fine.is_multiple_of(count) {
        return Err(format!("{}: mesh {count} does not divide the reference", case.id).into());
    }
    Ok(fine / count)
}

/// Encloses exact coarse cell averages as means of whole frozen 400-cell intervals.
///
/// Equal-width fine cells partition each coarse cell, so the exact coarse average is the
/// mean of the exact fine averages. Each endpoint mean widens by a bound on its rounding.
fn restricted_reference(case: Case, count: usize) -> TestResult<Vec<[f64; 4]>> {
    let fine = case.grids[2];
    let ratio = reference_ratio(case, count)?;
    Ok(fine
        .chunks_exact(ratio)
        .map(|block| {
            let mut cell = [0.0; 4];
            for (endpoint, value) in cell.iter_mut().enumerate() {
                let bits = block.iter().map(|row| f64::from_bits(row[endpoint]));
                let sum: f64 = bits.clone().sum();
                let magnitude: f64 = bits.map(f64::abs).sum();
                let bound = (ratio + 2) as f64 * f64::EPSILON * magnitude / ratio as f64
                    + f64::MIN_POSITIVE;
                let mean = sum / ratio as f64;
                *value = if endpoint % 2 == 0 {
                    mean - bound
                } else {
                    mean + bound
                };
            }
            cell[0] = cell[0].max(0.0);
            cell
        })
        .collect())
}

fn errors(case: Case, actual: &[[f64; 2]], reference: &[[f64; 4]]) -> TestResult<[f64; 2]> {
    if actual.len() != reference.len() {
        return Err("Surveyed reference grid length differs".into());
    }
    let mut total = [0.0; 2];
    for (actual, reference) in actual.iter().zip(reference) {
        for component in 0..2 {
            let lo = reference[2 * component];
            let hi = reference[2 * component + 1];
            let scale = f64::from_bits(case.scales[component]);
            assert!(scale.is_finite() && scale > 0.0);
            assert!(lo.is_finite() && hi.is_finite() && lo <= hi);
            assert!(hi - lo < 0.03 * scale / 100.0);
            assert!(actual[component].is_finite());
            // The furthest endpoint bounds error for every exact reference inside its interval.
            total[component] += (actual[component] - lo)
                .abs()
                .max((actual[component] - hi).abs())
                / scale;
        }
    }
    Ok(total.map(|value| value / actual.len() as f64))
}

/// Front observables of one run.
struct Fronts {
    /// Water in cells whose frozen 400-cell reference rows are all exactly zero, in m³.
    dry_volume: f64,
    /// Distance from each observed front to its exact front, in metres.
    distances: Vec<f64>,
}

fn fronts(case: Case, conserved: &[[f64; 2]]) -> TestResult<Fronts> {
    let count = conserved.len();
    let ratio = reference_ratio(case, count)?;
    let dx = 200.0 / count as f64;
    let dry_volume = conserved
        .iter()
        .zip(case.grids[2].chunks_exact(ratio))
        .filter(|(_, block)| block.iter().flatten().all(|&bits| bits == 0))
        .map(|(state, _)| state[0] * dx)
        .sum();
    // A cell is wet above the frozen threshold. The observed front is a cell edge.
    let threshold = f64::from_bits(case.scales[0]) * WET_FRACTION;
    let wet: Vec<_> = conserved.iter().map(|state| state[0] > threshold).collect();
    let center = count / 2;
    let mut distances = Vec::new();
    for front in case.fronts {
        let index = match (case.classification, front.direction < 0) {
            ("right_dry", _) => wet.iter().rposition(|&value| value).map(|index| index + 1),
            ("left_dry", _) => wet.iter().position(|&value| value),
            ("vacuum", true) => wet[..center]
                .iter()
                .rposition(|&value| value)
                .map(|index| index + 1),
            ("vacuum", false) => wet[center..]
                .iter()
                .position(|&value| value)
                .map(|index| index + center),
            _ => return Err(format!("{}: unsupported dry-front classification", case.id).into()),
        }
        .ok_or_else(|| format!("{}: dry front absent", case.id))?;
        let observed = -100.0 + index as f64 * dx;
        let distance = front
            .position
            .map(|bits| (observed - f64::from_bits(bits)).abs());
        distances.push(distance[0].max(distance[1]));
    }
    Ok(Fronts {
        dry_volume,
        distances,
    })
}

/// Requires dry-side water and every front distance to strictly decrease under refinement.
fn front_convergence(case: Case, measurements: &[Measurement]) -> TestResult {
    if case.fronts.is_empty() {
        return Ok(());
    }
    let mut observed = Vec::new();
    for measurement in measurements {
        let fronts = fronts(case, &measurement.conserved)?;
        println!(
            "surveyed {}: cells={}, dry_volume_m3={}, front_distance_m={:?}",
            case.id,
            measurement.conserved.len(),
            fronts.dry_volume,
            fronts.distances
        );
        observed.push(fronts);
    }
    decreasing(
        &format!("{} dry-side volume", case.id),
        &observed
            .iter()
            .map(|row| row.dry_volume)
            .collect::<Vec<_>>(),
    );
    for front in 0..case.fronts.len() {
        decreasing(
            &format!("{} front {front} distance", case.id),
            &observed
                .iter()
                .map(|row| row.distances[front])
                .collect::<Vec<_>>(),
        );
    }
    Ok(())
}

fn decreasing(label: &str, values: &[f64]) {
    assert!(
        values.windows(2).all(|pair| pair[1] < pair[0]),
        "{label}: {values:?}"
    );
}

fn differences(case: Case, coarse: &Measurement, fine: &Measurement) -> [f64; 2] {
    assert_eq!(fine.conserved.len() % coarse.conserved.len(), 0);
    let ratio = fine.conserved.len() / coarse.conserved.len();
    let mut total = [0.0; 2];
    for (coarse, fine) in coarse
        .conserved
        .iter()
        .zip(fine.conserved.chunks_exact(ratio))
    {
        for component in 0..2 {
            let restricted = fine.iter().map(|cell| cell[component]).sum::<f64>() / ratio as f64;
            total[component] +=
                (coarse[component] - restricted).abs() / f64::from_bits(case.scales[component]);
        }
    }
    total.map(|value| value / coarse.conserved.len() as f64)
}

fn spatial_convergence(case: Case) -> TestResult {
    assert_eq!(CASES.len(), 7);
    let mut measurements = Vec::new();
    for count in MESHES {
        measurements.push(run::<crate::runtime::Runtime>(
            crate::runtime::client(),
            case,
            count,
            SPATIAL_STEP_S,
        )?);
    }
    for component in 0..2 {
        let values: Vec<_> = measurements
            .iter()
            .map(|result| result.errors[component])
            .collect();
        decreasing(case.id, &values);
        assert!(
            values[2] < 0.03,
            "{} component={component}: {values:?}",
            case.id
        );
    }
    front_convergence(case, &measurements)
}

fn spatial_case(id: &str) -> TestResult<Case> {
    CASES
        .iter()
        .copied()
        .find(|case| case.id == id)
        .ok_or_else(|| {
            format!("Surveyed spatial family {id:?} absent: retain the frozen acceptance fixtures")
                .into()
        })
}

#[test]
fn trapezoid_wet_spatial_cell_averages_converge() -> TestResult {
    spatial_convergence(spatial_case("trapezoid_wet")?)
}

#[test]
fn reversed_trapezoid_wet_spatial_cell_averages_converge() -> TestResult {
    spatial_convergence(spatial_case("trapezoid_wet_reverse")?)
}

#[test]
fn multiband_wet_spatial_cell_averages_converge() -> TestResult {
    spatial_convergence(spatial_case("legacy_multiband")?)
}

#[test]
fn rounded_multiband_wet_fan_spatial_cell_averages_converge() -> TestResult {
    spatial_convergence(spatial_case("original_rounded_wet_fan")?)
}

#[test]
fn multiband_dry_fan_spatial_cell_averages_converge() -> TestResult {
    spatial_convergence(spatial_case("both_knots_dry_fan")?)
}

#[test]
fn reversed_multiband_dry_fan_spatial_cell_averages_converge() -> TestResult {
    spatial_convergence(spatial_case("both_knots_dry_fan_reverse")?)
}

#[test]
fn vacuum_spatial_cell_averages_converge() -> TestResult {
    spatial_convergence(spatial_case("generated_vacuum")?)
}

fn refinement_cases() -> impl Iterator<Item = Case> {
    CASES
        .iter()
        .copied()
        .filter(|case| matches!(case.id, "trapezoid_wet" | "both_knots_dry_fan"))
}

#[test]
fn fixed_spacing_temporal_conserved_differences_strictly_decrease() -> TestResult {
    for case in refinement_cases() {
        let mut measurements = Vec::new();
        for max_step in [0.5, 0.25, 0.125] {
            measurements.push(run::<crate::runtime::Runtime>(
                crate::runtime::client(),
                case,
                TEMPORAL_CELLS,
                max_step,
            )?);
        }
        let deltas: Vec<_> = measurements
            .windows(2)
            .map(|pair| differences(case, &pair[0], &pair[1]))
            .collect();
        println!("surveyed {} temporal differences: {deltas:?}", case.id);
        for component in 0..2 {
            decreasing(
                case.id,
                &deltas.iter().map(|row| row[component]).collect::<Vec<_>>(),
            );
        }
    }
    Ok(())
}

#[test]
fn combined_refinement_reduces_errors_and_restricted_conserved_differences() -> TestResult {
    for case in refinement_cases() {
        let mut measurements = Vec::new();
        for (count, max_step) in MESHES.into_iter().zip([0.25, 0.125, 0.0625]) {
            measurements.push(run::<crate::runtime::Runtime>(
                crate::runtime::client(),
                case,
                count,
                max_step,
            )?);
        }
        let deltas: Vec<_> = measurements
            .windows(2)
            .map(|pair| differences(case, &pair[0], &pair[1]))
            .collect();
        for component in 0..2 {
            decreasing(
                case.id,
                &measurements
                    .iter()
                    .map(|row| row.errors[component])
                    .collect::<Vec<_>>(),
            );
            decreasing(
                case.id,
                &deltas.iter().map(|row| row[component]).collect::<Vec<_>>(),
            );
            assert!(measurements[2].errors[component] < 0.03);
        }
        front_convergence(case, &measurements)?;
    }
    Ok(())
}

#[test]
fn exact_knot_rests_preserve_unequal_length_cells() -> TestResult {
    let case = CASES
        .iter()
        .find(|case| case.id == "both_knots_dry_fan")
        .ok_or("Missing original multiband knot profile")?;
    let lengths = [0.5, 2.0, 7.0, 1.25];
    for (area, depth) in [(2.0, 1.0), (7.0, 2.0)] {
        let current = reach::<crate::runtime::Runtime>(
            crate::runtime::client(),
            profile(case.id, case.points, case.cap),
            &lengths,
            &[0.0; 4],
            &[area; 4],
            &[0.0; 4],
            ChannelConfig {
                max_step: 0.01,
                ..ChannelConfig::default()
            },
            ChannelSurveyedConfig::default(),
        )?;
        let next = advance(current, 0.02, support::walls())?;
        for (cell, dx) in next.state()?.iter().zip(lengths) {
            support::near(cell.volume_m3, area * dx, 1e-12);
            support::near(cell.depth_m, depth, 3e-12);
            support::near(cell.longitudinal_momentum_m4_per_s, 0.0, 1e-10);
        }
    }
    Ok(())
}

fn representative_native_advancement<R>(client: R::Client) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: support::Client<R>,
{
    let case = *CASES.first().ok_or("Missing surveyed transport case")?;
    let result = run::<R>(client, case, NATIVE_CELLS, 0.5)?;
    assert!(result.conserved.iter().all(|state| state[0] > 0.0));
    assert!(result.conserved.iter().any(|state| state[1] != 0.0));
    Ok(())
}

#[test]
fn configured_cpu_client_advances_original_surveyed_transport() -> TestResult {
    representative_native_advancement::<crate::runtime::Runtime>(crate::runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn configured_cuda_client_advances_original_surveyed_transport() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    representative_native_advancement::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
