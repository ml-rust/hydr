//! Point geometry against the frozen reference rows and the dense sweep.

use super::dense_rows::DENSE;
use super::reference_rows::{BARRELS, REFERENCE_SHA256, ROWS, State};
use super::support::{
    QUANTITIES, TestResult, barrel, batch, contains, geometry, quantities, state,
};
use hydr::structures::{BarrelState, CulvertBarrel, CulvertBarrelShape};

#[test]
fn reference_fixture_is_the_frozen_file() {
    assert_eq!(
        REFERENCE_SHA256,
        "8fb069d700a1800d379203dd541268e9ab573cde6731129840be3835843811e5"
    );
    assert_eq!(ROWS.len(), BARRELS.len() * 17);
}

#[test]
fn reference_rows_lie_inside_enclosures_within_frozen_widths() -> TestResult {
    // One row per reference row: circles and boxes interleave in one batch.
    let barrels = ROWS
        .iter()
        .map(|row| barrel(&BARRELS[row.barrel]))
        .collect::<Result<Vec<_>, _>>()?;
    let depths: Vec<f64> = ROWS.iter().map(|row| f64::from_bits(row.depth)).collect();
    let rows = geometry(&batch(barrels)?, &depths)?;
    for (reference, row) in ROWS.iter().zip(&rows) {
        let name = BARRELS[reference.barrel].name;
        let fraction = reference.fraction;
        assert_eq!(row.state, state(reference.state), "{name} at {fraction}");
        for (q, (bounds, (exact, allowance))) in quantities(row)
            .into_iter()
            .zip(reference.values.iter().zip(reference.relative_width))
            .enumerate()
        {
            let quantity = QUANTITIES[q];
            assert!(
                contains(bounds, *exact),
                "{name} at {fraction}: {quantity} {bounds:?} misses the reference"
            );
            let scale = f64::from_bits(exact[0])
                .abs()
                .max(f64::from_bits(exact[1]).abs());
            let width = bounds.1 - bounds.0;
            assert!(
                width <= allowance * scale,
                "{name} at {fraction}: {quantity} width {width:e} exceeds {allowance:e} of {scale:e}"
            );
        }
    }
    Ok(())
}

#[test]
fn dry_rows_are_exact_zeros_with_box_edges() -> TestResult {
    let barrels = BARRELS.iter().map(barrel).collect::<Result<Vec<_>, _>>()?;
    let rows = geometry(&batch(barrels.clone())?, &vec![0.0; barrels.len()])?;
    for (barrel, row) in barrels.iter().zip(&rows) {
        assert_eq!(row.state, BarrelState::Dry);
        let expected_edge = match barrel.shape() {
            CulvertBarrelShape::Circular => 0.0,
            _ => barrel.span_m(),
        };
        assert_eq!(row.area_m2, (0.0, 0.0));
        assert_eq!(row.top_width_m, (expected_edge, expected_edge));
        assert_eq!(row.wetted_perimeter_m, (expected_edge, expected_edge));
        assert_eq!(row.hydraulic_radius_m, (0.0, 0.0));
        assert_eq!(row.hydrostatic_moment_m3, (0.0, 0.0));
    }
    Ok(())
}

#[test]
fn subnormal_depths_enclose_without_rejection() -> TestResult {
    let barrels = vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::rectangular_box(2.0, 1.0)?,
    ];
    let depths = [f64::from_bits(1), 1e-300, f64::from_bits(1)];
    let rows = geometry(&batch(barrels)?, &depths)?;
    for (row, depth) in rows.iter().zip(depths) {
        assert_eq!(row.state, BarrelState::Free);
        for (lower, upper) in quantities(row) {
            assert!(
                0.0 <= lower && lower <= upper && upper.is_finite(),
                "{depth:e}"
            );
        }
        // 0 < R <= y: the series ratio keeps an upper bound near the depth.
        assert!(row.hydraulic_radius_m.1 <= 2.0 * depth.max(f64::from_bits(2)));
    }
    assert!(rows[1].wetted_perimeter_m.0 > 0.0);
    Ok(())
}

#[test]
fn dense_sweep_contains_references_without_certified_reversal() -> TestResult {
    for (entry, references) in DENSE {
        let barrels = vec![barrel(entry)?; references.len()];
        let depths: Vec<f64> = references.iter().map(|r| f64::from_bits(r.depth)).collect();
        let rows = geometry(&batch(barrels)?, &depths)?;
        for (reference, row) in references.iter().zip(&rows) {
            let depth = f64::from_bits(reference.depth);
            assert_eq!(
                row.state,
                state(reference.state),
                "{} at {depth}",
                entry.name
            );
            for (q, (bounds, exact)) in quantities(row)
                .into_iter()
                .zip(reference.values)
                .enumerate()
            {
                assert!(
                    contains(bounds, exact),
                    "{} at {depth}: {} {bounds:?} misses the reference",
                    entry.name,
                    QUANTITIES[q]
                );
            }
        }
        assert_eq!(references.last().map(|r| r.state), Some(State::Full));
        // A, P and M increase in depth: no later upper bound lies below an earlier lower bound.
        for pair in rows.windows(2) {
            for q in [0, 2, 4] {
                let (before, after) = (quantities(&pair[0])[q], quantities(&pair[1])[q]);
                assert!(
                    after.1 >= before.0,
                    "{}: {} reverses",
                    entry.name,
                    QUANTITIES[q]
                );
            }
        }
    }
    Ok(())
}
