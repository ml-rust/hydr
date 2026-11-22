//! Arc barrel sweeps: monotone depth sweeps, `dA/dy = T` and `dM/dy = A`, and depth ranges.

use super::geometry::{Outcome, copies, points, quantities, tensor};
use super::reference_rows::{BARRELS, ROWS, State};
use super::{TestResult, barrel};
use hydr::structures::{StructureError, StructureInputErrorKind};

/// Depth fractions of the `dA/dy` checks, away from the floor and the crown.
const SLOPE_FRACTIONS: [f64; 8] = [1e-3, 0.03, 0.1, 0.2, 0.37, 0.61, 0.83, 0.97];

/// Relative slack of the slope checks: the curvature of `T` and `A` over the step.
const SLOPE_SLACK: f64 = 1e-9;

/// Every free depth of evidence barrel `index` in the frozen rows.
fn free_depths(index: usize) -> Vec<f64> {
    ROWS.iter()
        .filter(|row| row.barrel == index && row.state == State::Free)
        .map(|row| f64::from_bits(row.depth))
        .collect()
}

#[test]
fn dense_sweeps_hold_no_certified_reversal() -> TestResult {
    for (index, entry) in BARRELS.iter().enumerate() {
        let rise = barrel(entry)?.rise_m();
        let mut depths: Vec<f64> = (0..=256).map(|k| rise * f64::from(k) / 256.0).collect();
        depths.extend((1..=40).map(|k| rise * 2f64.powi(-k)));
        depths.extend(free_depths(index));
        depths.sort_by(f64::total_cmp);
        depths.dedup();
        let rows = points(index, &depths)?;
        for (pair, window) in rows.windows(2).zip(depths.windows(2)) {
            let [low, high] = pair else { continue };
            let [from, to] = window else { continue };
            let [a0, _, p0, _, m0] = quantities(low);
            let [a1, _, p1, _, m1] = quantities(high);
            for (name, (before, after)) in
                ["area", "perimeter", "moment"]
                    .into_iter()
                    .zip([(a0, a1), (p0, p1), (m0, m1)])
            {
                assert!(
                    after.1 >= before.0,
                    "{} {name} falls from {from:e} to {to:e}: {before:?} then {after:?}",
                    entry.name
                );
            }
        }
    }
    Ok(())
}

/// Requires the difference quotient of `(left, right)` over `2 h` to meet the hull of `slopes`.
fn quotient_meets(
    what: &str,
    left: (f64, f64),
    right: (f64, f64),
    step: f64,
    slopes: [(f64, f64); 3],
) -> Result<(), String> {
    let low = (right.0 - left.1) / (2.0 * step);
    let high = (right.1 - left.0) / (2.0 * step);
    let floor = slopes.iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
    let ceiling = slopes.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max);
    let slack = SLOPE_SLACK * ceiling.abs();
    if low <= ceiling + slack && floor - slack <= high {
        Ok(())
    } else {
        Err(format!(
            "{what}: quotient [{low:e}, {high:e}] misses [{floor:e}, {ceiling:e}]"
        ))
    }
}

#[test]
fn centred_differences_of_area_and_moment_meet_width_and_area() -> TestResult {
    let mut found = Vec::new();
    for (index, entry) in BARRELS.iter().enumerate() {
        let rise = barrel(entry)?.rise_m();
        let step = rise * 2f64.powi(-16);
        let mut centres: Vec<f64> = SLOPE_FRACTIONS.iter().map(|s| s * rise).collect();
        centres.extend(
            free_depths(index)
                .into_iter()
                .filter(|y| *y > 2.0 * step && *y < rise - 2.0 * step),
        );
        let depths: Vec<f64> = centres
            .iter()
            .flat_map(|y| [y - step, *y, y + step])
            .collect();
        let rows = points(index, &depths)?;
        for (centre, [left, middle, right]) in centres.iter().zip(rows.as_chunks::<3>().0) {
            let what = format!("{} at {centre:e}", entry.name);
            let widths = [left.top_width_m, middle.top_width_m, right.top_width_m];
            let areas = [left.area_m2, middle.area_m2, right.area_m2];
            let checks = [
                quotient_meets(
                    &format!("{what} dA/dy"),
                    left.area_m2,
                    right.area_m2,
                    step,
                    widths,
                ),
                quotient_meets(
                    &format!("{what} dM/dy"),
                    left.hydrostatic_moment_m3,
                    right.hydrostatic_moment_m3,
                    step,
                    areas,
                ),
            ];
            found.extend(checks.into_iter().filter_map(Result::err));
        }
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
    Ok(())
}

#[test]
fn ranges_hold_their_end_enclosures_and_every_reference_they_span() -> TestResult {
    for (index, entry) in BARRELS.iter().enumerate() {
        // Every pair of rows from the dry floor up to the last free row, in depth order.
        let rows: Vec<_> = ROWS
            .iter()
            .filter(|row| row.barrel == index && row.state != State::Full)
            .collect();
        let mut pairs = Vec::new();
        for (k, low) in rows.iter().enumerate() {
            for high in rows.iter().skip(k) {
                pairs.push((low, high));
            }
        }
        let lower: Vec<f64> = pairs.iter().map(|(l, _)| f64::from_bits(l.depth)).collect();
        let upper: Vec<f64> = pairs.iter().map(|(_, h)| f64::from_bits(h.depth)).collect();
        let batch = copies(index, pairs.len())?;
        let spans = batch
            .geometry_over(&tensor(&lower)?, &tensor(&upper)?)?
            .project()?;
        let lows = batch.geometry(&tensor(&lower)?)?.project()?;
        let highs = batch.geometry(&tensor(&upper)?)?.project()?;
        for (((span, low), high), (l, h)) in spans.iter().zip(&lows).zip(&highs).zip(&pairs) {
            let label = format!("{} range {} to {}", entry.name, l.fraction, h.fraction);
            // R divides two ranges, so it holds the exact values but not each end enclosure.
            for end in [low, high] {
                let held = quantities(span).into_iter().zip(quantities(end));
                for (q, (range, value)) in held.enumerate().filter(|(q, _)| *q != 3) {
                    assert!(
                        range.0 <= value.0 && value.1 <= range.1,
                        "{label} quantity {q}: {range:?} misses the end {value:?}"
                    );
                }
            }
            let spanned = rows
                .iter()
                .filter(|row| row.depth >= l.depth && row.depth <= h.depth);
            for row in spanned {
                for (q, (range, exact)) in quantities(span).into_iter().zip(row.values).enumerate()
                {
                    let [down, up] = exact.map(f64::from_bits);
                    assert!(
                        range.0 <= down && up <= range.1,
                        "{label} quantity {q}: {range:?} misses {} [{down:e}, {up:e}]",
                        row.fraction
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn ranges_across_the_widest_depth_reach_the_widest_width() -> TestResult {
    for (index, entry) in BARRELS.iter().enumerate() {
        let Some(widest) = ROWS
            .iter()
            .find(|row| row.barrel == index && (row.fraction == "widest" || row.fraction == "1/2"))
            .map(|row| f64::from_bits(row.depth))
        else {
            continue;
        };
        let batch = copies(index, 1)?;
        let rise = barrel(entry)?.rise_m();
        let low = 0.5 * widest;
        let high = 0.5 * (widest + rise);
        let span = batch
            .geometry_over(&tensor(&[low])?, &tensor(&[high])?)?
            .project()?;
        let point = batch.geometry(&tensor(&[widest])?)?.project()?;
        let (Some(span), Some(point)) = (span.first(), point.first()) else {
            return Err("one row expected".into());
        };
        assert!(
            span.top_width_m.1 >= point.top_width_m.1,
            "{}: range width {:?} below the widest {:?}",
            entry.name,
            span.top_width_m,
            point.top_width_m
        );
    }
    Ok(())
}

#[test]
fn ranges_reaching_the_crown_reject() -> Outcome<()> {
    for (index, entry) in BARRELS.iter().enumerate() {
        let rise = barrel(entry)?.rise_m();
        let batch = copies(index, 1)?;
        match batch.geometry_over(&tensor(&[0.0])?, &tensor(&[rise])?) {
            Err(StructureError::InvalidInput {
                kind: StructureInputErrorKind::OperatingPoint,
                context,
                ..
            }) if context.contains("the range reaches the crown") => {}
            Err(other) => return Err(format!("{}: {other:?}", entry.name).into()),
            Ok(_) => return Err(format!("{}: a crown range was admitted", entry.name).into()),
        }
    }
    Ok(())
}
