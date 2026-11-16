use super::common::{GRAVITY, TestResult, advance, cell, definition, exterior, reach};
use crate::triangle_reference::{dry_release, exact_wetshock, mean_errors, twofan_vacuum};
use hydr::channel::ChannelTriangularReconstruction;

#[derive(Clone, Copy, Debug)]
enum Case {
    DryRelease,
    WetShock,
    Vacuum,
}

impl Case {
    fn reference(self, left: f64, right: f64, time: f64) -> (f64, f64) {
        match self {
            Self::DryRelease => dry_release(left, right, time, GRAVITY, 1.0),
            Self::WetShock => exact_wetshock(left, right, time, GRAVITY),
            Self::Vacuum => twofan_vacuum(left, right, time, GRAVITY, 1.0),
        }
    }
    fn time(self) -> f64 {
        match self {
            Self::Vacuum => 1.0,
            _ => 4.0,
        }
    }
    /// Fixed interval below the 400-cell CFL limit. Every spatial level takes the same steps.
    fn spatial_step(self) -> f64 {
        match self {
            Self::DryRelease => 0.008,
            Self::WetShock => 0.006,
            Self::Vacuum => 0.003,
        }
    }
    fn boundaries(self) -> [hydr::channel::ChannelBoundary; 2] {
        match self {
            Self::DryRelease => [exterior(1.0, 0.0), exterior(0.0, 0.0)],
            Self::WetShock => [
                exterior(2.0, (7.0 * GRAVITY / 4.0).sqrt()),
                exterior(1.0, 0.0),
            ],
            Self::Vacuum => {
                let velocity = 5.0 * (GRAVITY / 2.0).sqrt();
                [exterior(1.0, -velocity), exterior(1.0, velocity)]
            }
        }
    }
}

struct ResultState {
    errors: (f64, f64),
    conserved: Vec<(f64, f64)>,
}

fn measure(
    case: Case,
    count: usize,
    max_step: f64,
    reconstruction: ChannelTriangularReconstruction,
) -> Result<ResultState, Box<dyn std::error::Error>> {
    let dx = 200.0 / count as f64;
    let intervals: Vec<_> = (0..count)
        .map(|i| {
            let left = -100.0 + i as f64 * dx;
            (left, left + dx)
        })
        .collect();
    let initial: Vec<_> = intervals
        .iter()
        .map(|&(left, right)| case.reference(left, right, 0.0))
        .collect();
    let areas: Vec<_> = initial.iter().map(|state| state.0).collect();
    let discharges: Vec<_> = initial.iter().map(|state| state.1).collect();
    let mut layout = definition((0..count).map(|i| cell(i, dx)).collect(), max_step);
    layout.config.triangular_reconstruction = reconstruction;
    let initial = reach(layout, &areas, &discharges)?;
    let result = advance(initial, case.time(), case.boundaries())?;
    let conserved: Vec<_> = result
        .state()?
        .iter()
        .map(|cell| {
            (
                cell.volume_m3 / dx,
                cell.longitudinal_momentum_m4_per_s / dx,
            )
        })
        .collect();
    let errors = mean_errors(
        conserved
            .iter()
            .zip(&intervals)
            .map(|(&(area, q), &(left, right))| {
                let reference = case.reference(left, right, case.time());
                (dx, area, q, reference.0, reference.1)
            }),
        1.0,
    )
    .ok_or("Missing triangular error norm")?;
    println!(
        "triangle {case:?} {reconstruction:?}: cells={count}, max_step={max_step}, mean_abs_h={}, mean_abs_Q={}",
        errors.0, errors.1
    );
    Ok(ResultState { errors, conserved })
}

fn run(case: Case, count: usize, max_step: f64) -> Result<ResultState, Box<dyn std::error::Error>> {
    assert_eq!(
        hydr::channel::ChannelConfig::default().triangular_reconstruction,
        ChannelTriangularReconstruction::LimitedLinear
    );
    let result = measure(
        case,
        count,
        max_step,
        ChannelTriangularReconstruction::LimitedLinear,
    )?;
    // Frozen before execution. Reference depth comes from independently averaged conserved area.
    assert!(
        result.errors.0 < 0.06,
        "{case:?}: depth error {}",
        result.errors.0
    );
    assert!(
        result.errors.1 < 0.3,
        "{case:?}: discharge error {}",
        result.errors.1
    );
    Ok(result)
}

fn decreasing(label: &str, values: &[f64]) {
    assert!(
        values.windows(2).all(|pair| pair[1] < pair[0]),
        "{label}: {values:?}"
    );
}

// Restrict fine conserved averages before recovering their represented depth.
fn differences(coarse: &ResultState, fine: &ResultState) -> (f64, f64) {
    assert_eq!(fine.conserved.len() % coarse.conserved.len(), 0);
    let ratio = fine.conserved.len() / coarse.conserved.len();
    let mut depth = 0.0;
    let mut discharge = 0.0;
    for (&(area, q), fine_cells) in coarse
        .conserved
        .iter()
        .zip(fine.conserved.chunks_exact(ratio))
    {
        let fine_area = fine_cells.iter().map(|state| state.0).sum::<f64>() / ratio as f64;
        let fine_q = fine_cells.iter().map(|state| state.1).sum::<f64>() / ratio as f64;
        depth += (area.sqrt() - fine_area.sqrt()).abs();
        discharge += (q - fine_q).abs();
    }
    (
        depth / coarse.conserved.len() as f64,
        discharge / coarse.conserved.len() as f64,
    )
}

#[test]
fn exact_conserved_cell_averages_converge_under_spatial_refinement() -> TestResult {
    for case in [Case::DryRelease, Case::WetShock, Case::Vacuum] {
        let mut depth = Vec::new();
        let mut discharge = Vec::new();
        for count in [100, 200, 400] {
            let result = run(case, count, case.spatial_step())?;
            depth.push(result.errors.0);
            discharge.push(result.errors.1);
        }
        decreasing("Spatial depth", &depth);
        decreasing("Spatial discharge", &discharge);
    }
    Ok(())
}

#[test]
fn dry_release_temporal_differences_decrease_at_fixed_spacing() -> TestResult {
    let mut results = Vec::new();
    for max_step in [0.02, 0.01, 0.005] {
        results.push(run(Case::DryRelease, 50, max_step)?);
    }
    let differences: Vec<_> = results
        .windows(2)
        .map(|pair| differences(&pair[0], &pair[1]))
        .collect();
    println!("triangle temporal successive depth/discharge differences: {differences:?}");
    decreasing(
        "Temporal depth difference",
        &differences.iter().map(|value| value.0).collect::<Vec<_>>(),
    );
    decreasing(
        "Temporal discharge difference",
        &differences.iter().map(|value| value.1).collect::<Vec<_>>(),
    );
    Ok(())
}

#[test]
fn dry_release_combined_refinement_reduces_errors_and_successive_differences() -> TestResult {
    let mut results = Vec::new();
    for (count, max_step) in [(100, 0.02), (200, 0.01), (400, 0.005)] {
        results.push(run(Case::DryRelease, count, max_step)?);
    }
    decreasing(
        "Combined depth",
        &results
            .iter()
            .map(|result| result.errors.0)
            .collect::<Vec<_>>(),
    );
    decreasing(
        "Combined discharge",
        &results
            .iter()
            .map(|result| result.errors.1)
            .collect::<Vec<_>>(),
    );
    let differences: Vec<_> = results
        .windows(2)
        .map(|pair| differences(&pair[0], &pair[1]))
        .collect();
    println!("triangle combined successive depth/discharge differences: {differences:?}");
    decreasing(
        "Combined depth difference",
        &differences.iter().map(|value| value.0).collect::<Vec<_>>(),
    );
    decreasing(
        "Combined discharge difference",
        &differences.iter().map(|value| value.1).collect::<Vec<_>>(),
    );
    Ok(())
}

#[test]
fn first_order_reference_vacuum_errors_decrease_without_production_ceiling_claims() -> TestResult {
    let mut depth = Vec::new();
    let mut discharge = Vec::new();
    for count in [100, 200, 400] {
        let result = measure(
            Case::Vacuum,
            count,
            Case::Vacuum.spatial_step(),
            ChannelTriangularReconstruction::FirstOrderReference,
        )?;
        assert!(result.errors.0.is_finite() && result.errors.0 >= 0.0);
        assert!(result.errors.1.is_finite() && result.errors.1 >= 0.0);
        depth.push(result.errors.0);
        discharge.push(result.errors.1);
    }
    decreasing("Reference vacuum depth", &depth);
    decreasing("Reference vacuum discharge", &discharge);
    Ok(())
}
