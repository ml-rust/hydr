//! Arc barrel point geometry against the frozen references, the dry floor and the crown.

use super::reference_rows::{BARRELS, ROWS, Row, Shape, State};
use super::runtime::{Runtime, client};
use super::{TestResult, barrel};
use hydr::structures::{BarrelState, CulvertBarrelBatch, CulvertBarrelGeometryRow};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

/// Quantity names in fixture order.
pub const QUANTITIES: [&str; 5] = ["area", "top_width", "perimeter", "radius", "moment"];

pub type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

pub fn tensor(values: &[f64]) -> Result<Tensor<Runtime>, numr::error::Error> {
    Tensor::from_slice(values, &[values.len()], client().device())
}

/// `(lower, upper)` of each quantity in fixture order.
pub fn quantities(row: &CulvertBarrelGeometryRow) -> [(f64, f64); 5] {
    [
        row.area_m2,
        row.top_width_m,
        row.wetted_perimeter_m,
        row.hydraulic_radius_m,
        row.hydrostatic_moment_m3,
    ]
}

/// A batch of `count` copies of evidence barrel `index`.
pub fn copies(index: usize, count: usize) -> Outcome<CulvertBarrelBatch<Runtime>> {
    let entry = BARRELS.get(index).ok_or("no such evidence barrel")?;
    let one = barrel(entry)?;
    Ok(CulvertBarrelBatch::new(client(), vec![one; count])?)
}

/// Point geometry of evidence barrel `index` at `depths`.
pub fn points(index: usize, depths: &[f64]) -> Outcome<Vec<CulvertBarrelGeometryRow>> {
    let batch = copies(index, depths.len())?;
    Ok(batch.geometry(&tensor(depths)?)?.project()?)
}

fn state(value: State) -> BarrelState {
    match value {
        State::Dry => BarrelState::Dry,
        State::Free => BarrelState::Free,
        State::Full => BarrelState::Full,
    }
}

/// Every failure of `row` against its reference, as text.
fn failures(reference: &Row, row: &CulvertBarrelGeometryRow) -> Vec<String> {
    let name = BARRELS.get(reference.barrel).map_or("?", |b| b.name);
    let fraction = reference.fraction;
    let mut found = Vec::new();
    if row.state != state(reference.state) {
        found.push(format!("{name} at {fraction}: state {:?}", row.state));
    }
    let pairs = quantities(row)
        .into_iter()
        .zip(reference.values.iter().zip(reference.relative_width));
    for ((bounds, (exact, allowance)), quantity) in pairs.zip(QUANTITIES) {
        let [down, up] = exact.map(f64::from_bits);
        if !(bounds.0 <= down && up <= bounds.1) {
            found.push(format!(
                "{name} at {fraction}: {quantity} {bounds:?} misses [{down:e}, {up:e}]"
            ));
        }
        let width = bounds.1 - bounds.0;
        let scale = down.abs().max(up.abs());
        if width > allowance * scale {
            found.push(format!(
                "{name} at {fraction}: {quantity} width {width:e} exceeds {allowance:e} of {scale:e}"
            ));
        }
    }
    found
}

#[test]
fn every_reference_row_lies_inside_its_enclosure_within_the_frozen_width() -> TestResult {
    // All rows of every barrel in one batch, so arc rows of all four shapes interleave.
    let barrels = ROWS
        .iter()
        .map(|row| BARRELS.get(row.barrel).ok_or("row names no barrel"))
        .map(|entry| barrel(entry?))
        .collect::<Outcome<Vec<_>>>()?;
    let depths: Vec<f64> = ROWS.iter().map(|row| f64::from_bits(row.depth)).collect();
    let batch = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?;
    let rows = batch.geometry(&tensor(&depths)?)?.project()?;
    let found: Vec<String> = ROWS
        .iter()
        .zip(&rows)
        .flat_map(|(reference, row)| failures(reference, row))
        .collect();
    assert!(found.is_empty(), "{}", found.join("\n"));
    Ok(())
}

#[test]
fn dry_floor_is_exact_with_the_arch_floor_wetted() -> TestResult {
    for (index, entry) in BARRELS.iter().enumerate() {
        let built = barrel(entry)?;
        let rows = points(index, &[0.0])?;
        let [row] = rows.as_slice() else {
            return Err("one row expected".into());
        };
        let edge = if entry.shape == Shape::Arch {
            built.span_m()
        } else {
            0.0
        };
        assert_eq!(row.state, BarrelState::Dry, "{}", entry.name);
        assert_eq!(row.area_m2, (0.0, 0.0), "{}", entry.name);
        assert_eq!(row.top_width_m, (edge, edge), "{}", entry.name);
        assert_eq!(row.wetted_perimeter_m, (edge, edge), "{}", entry.name);
        assert_eq!(row.hydraulic_radius_m, (0.0, 0.0), "{}", entry.name);
        assert_eq!(row.hydrostatic_moment_m3, (0.0, 0.0), "{}", entry.name);
    }
    Ok(())
}

#[test]
fn crown_closes_onto_the_full_state() -> TestResult {
    for (index, entry) in BARRELS.iter().enumerate() {
        let rise = barrel(entry)?.rise_m();
        let rows = points(index, &[rise.next_down(), rise, 2.0 * rise])?;
        let [free, crown, above] = rows.as_slice() else {
            return Err("three rows expected".into());
        };
        let name = entry.name;
        assert_eq!(free.state, BarrelState::Free, "{name}");
        assert_eq!(crown.state, BarrelState::Full, "{name}");
        assert_eq!(above.state, BarrelState::Full, "{name}");
        assert_eq!(crown.top_width_m, (0.0, 0.0), "{name}");
        // One ulp below the crown the free area and perimeter sit just below the full values.
        let gap = rise - rise.next_down();
        for (low, full, slack) in [
            (free.area_m2, crown.area_m2, 4.0 * rise * gap),
            (
                free.wetted_perimeter_m,
                crown.wetted_perimeter_m,
                64.0 * (rise * gap).sqrt(),
            ),
            (
                free.hydrostatic_moment_m3,
                crown.hydrostatic_moment_m3,
                4.0 * rise * rise * gap,
            ),
        ] {
            assert!(low.0 <= full.1, "{name}: free {low:?} above full {full:?}");
            assert!(full.0 - low.1 <= slack, "{name}: gap {low:?} to {full:?}");
        }
        assert!(free.top_width_m.1 <= 64.0 * (rise * gap).sqrt(), "{name}");
        // Above the crown the moment grows by A_full times the head.
        let head = rise;
        let grown = (
            above.hydrostatic_moment_m3.0 - crown.hydrostatic_moment_m3.1,
            above.hydrostatic_moment_m3.1 - crown.hydrostatic_moment_m3.0,
        );
        let expected = (crown.area_m2.0 * head, crown.area_m2.1 * head);
        assert!(
            grown.0 <= expected.1 * (1.0 + 1e-12) && expected.0 <= grown.1 * (1.0 + 1e-12),
            "{name}: moment rise {grown:?} against {expected:?}"
        );
        assert_eq!(above.area_m2, crown.area_m2, "{name}");
        assert_eq!(above.wetted_perimeter_m, crown.wetted_perimeter_m, "{name}");
    }
    Ok(())
}

#[test]
fn full_geometry_reports_the_crown_moment() -> TestResult {
    for (index, entry) in BARRELS.iter().enumerate() {
        let rise = barrel(entry)?.rise_m();
        let batch = copies(index, 1)?;
        let full = batch.full_geometry()?.project()?;
        let crown = batch.geometry(&tensor(&[rise])?)?.project()?;
        assert_eq!(full, crown, "{}", entry.name);
    }
    Ok(())
}
