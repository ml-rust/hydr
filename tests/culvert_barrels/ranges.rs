//! Range geometry over `[lower, upper]` inside `[0, D)`.

use super::dense_rows::DENSE;
use super::support::{
    QUANTITIES, TestResult, barrel, batch, contains, geometry, quantities, tensor,
};
use hydr::structures::{BarrelState, CulvertBarrelShape, StructureError, StructureInputErrorKind};

/// Dense-sweep index pairs. Index 64 is `D/16`, 182 the first past `D/2`, 255 the last free depth.
const RANGES: [(usize, usize); 6] = [(0, 10), (3, 63), (60, 70), (100, 220), (190, 255), (0, 255)];

#[test]
fn ranges_contain_the_point_enclosures_they_span() -> TestResult {
    for (entry, references) in DENSE {
        let depth = |i: usize| f64::from_bits(references[i].depth);
        let culvert = barrel(entry)?;
        let ranges = batch(vec![culvert; RANGES.len()])?;
        let lower: Vec<f64> = RANGES.iter().map(|&(i, _)| depth(i)).collect();
        let upper: Vec<f64> = RANGES.iter().map(|&(_, j)| depth(j)).collect();
        let spans = ranges
            .geometry_over(&tensor(&lower)?, &tensor(&upper)?)?
            .project()?;
        let points = batch(vec![culvert; references.len()])?;
        let all: Vec<f64> = (0..references.len()).map(depth).collect();
        let point_rows = geometry(&points, &all)?;
        for (&(first, last), span) in RANGES.iter().zip(&spans) {
            let expected = if depth(last) > 0.0 {
                BarrelState::Free
            } else {
                BarrelState::Dry
            };
            assert_eq!(span.state, expected);
            let bounds = quantities(span);
            let spanned = point_rows[first..=last]
                .iter()
                .zip(&references[first..=last]);
            for (k, (point_row, reference)) in (first..).zip(spanned) {
                let point = quantities(point_row);
                for (q, ((range, inner), exact)) in
                    bounds.iter().zip(point).zip(reference.values).enumerate()
                {
                    let label =
                        format!("{} [{first}, {last}] at {k}: {}", entry.name, QUANTITIES[q]);
                    assert!(contains(*range, exact), "{label} misses the reference");
                    // R divides two ranges, so only the exact value is checked for it.
                    if QUANTITIES[q] != "radius" {
                        assert!(
                            range.0 <= inner.0 && inner.1 <= range.1,
                            "{label}: {inner:?} escapes {range:?}"
                        );
                    }
                }
            }
        }
        // A circle range holding D/2 reaches the diameter.
        if culvert.shape() == CulvertBarrelShape::Circular {
            assert_eq!(spans[3].top_width_m.1, culvert.rise_m());
        }
    }
    Ok(())
}

#[test]
fn ranges_reaching_the_crown_or_reversed_reject_with_their_row() -> TestResult {
    let (entry, _) = &DENSE[0];
    let culvert = barrel(entry)?;
    let rise = culvert.rise_m();
    let ranges = batch(vec![culvert; 2])?;
    let cases = [
        (
            [0.1, 0.2],
            [0.3, rise],
            1,
            "the range reaches the crown. Split it at the crown",
        ),
        (
            [0.4, 0.2],
            [0.3, 0.5],
            0,
            "lower_m=0.4,upper_m=0.3: supply lower_m <= upper_m",
        ),
        (
            [0.1, -0.2],
            [0.3, 0.5],
            1,
            "lower_m=-0.2: supply a nonnegative depth",
        ),
    ];
    for (lower, upper, row, message) in cases {
        let result = ranges.geometry_over(&tensor(&lower)?, &tensor(&upper)?);
        assert!(
            matches!(
                &result,
                Err(StructureError::InvalidInput {
                    element: Some(element),
                    kind: StructureInputErrorKind::OperatingPoint,
                    context,
                }) if *element == row && context.contains(message)
            ),
            "{message}: {:?}",
            result.err()
        );
    }
    Ok(())
}
