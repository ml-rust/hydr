//! Critical depth against the frozen independent roots, with typed outcomes and monotonicity.

use super::critical_rows::{CRITICAL, CRITICAL_SHA256, Class, GRAVITY};
use super::reference_rows::{BARRELS, REFERENCE_SHA256};
use super::support::{TestResult, barrel, batch, tensor};
use hydr::structures::{CriticalDepthStatus, CulvertBarrel, CulvertBarrelDepthRow};

type Row = CulvertBarrelDepthRow<CriticalDepthStatus>;

fn critical(
    barrels: Vec<CulvertBarrel>,
    discharges: &[f64],
    gravity: f64,
) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    Ok(batch(barrels)?
        .critical_depth(&tensor(discharges)?, gravity)?
        .project()?)
}

fn holds(depth: Option<(f64, f64)>, root: [u64; 2]) -> bool {
    depth.is_some_and(|(lower, upper)| {
        lower <= f64::from_bits(root[0]) && f64::from_bits(root[1]) <= upper
    })
}

#[test]
fn critical_fixture_is_the_frozen_file() {
    assert_eq!(CRITICAL_SHA256, REFERENCE_SHA256);
    assert_eq!(CRITICAL.len(), 3 * 10 + 3 * 8);
}

#[test]
fn reference_roots_lie_inside_enclosures_with_matching_outcomes() -> TestResult {
    use CriticalDepthStatus::{AboveCrown, AtCrown, BelowFloor, Dry, Solved};
    // One row per reference root: circles and boxes interleave in one batch.
    let barrels = CRITICAL
        .iter()
        .map(|row| barrel(&BARRELS[row.barrel]))
        .collect::<Result<Vec<_>, _>>()?;
    let discharges: Vec<f64> = CRITICAL
        .iter()
        .map(|row| f64::from_bits(row.discharge))
        .collect();
    let rows = critical(barrels, &discharges, f64::from_bits(GRAVITY))?;
    for (reference, row) in CRITICAL.iter().zip(&rows) {
        let entry = &BARRELS[reference.barrel];
        let rise = f64::from_bits(entry.rise);
        let what = format!("{} at {}: {row:?}", entry.name, reference.label);
        let admitted: &[CriticalDepthStatus] = match reference.class {
            Class::Dry => &[Dry],
            Class::BelowFloor => &[BelowFloor],
            Class::NearFloor => &[BelowFloor, Solved],
            Class::Free => &[Solved],
            Class::NearCrown => &[Solved, AtCrown],
            Class::NearCrownAbsent => &[AtCrown, AboveCrown],
            Class::AboveCrown => &[AboveCrown],
        };
        assert!(admitted.contains(&row.status), "{what}");
        if let Some(root) = reference.root {
            assert!(holds(row.depth_m, root), "{what} misses {root:x?}");
        }
        match (row.status, row.depth_m) {
            (Dry, depth) => assert_eq!(depth, Some((0.0, 0.0)), "{what}"),
            (BelowFloor, Some((lower, upper))) => {
                assert_eq!(lower, 0.0, "{what}");
                assert!(upper <= 2.0 * rise * 2f64.powi(-30), "{what}");
            }
            (Solved, Some((lower, upper))) if reference.class == Class::Free => {
                let scale = f64::from_bits(reference.root.map_or(0, |r| r[1]));
                let allowance = reference.relative_width * scale;
                assert!(
                    upper - lower <= allowance,
                    "{what} wider than {allowance:e}"
                );
            }
            (AtCrown, Some((_, upper))) => assert_eq!(upper, rise, "{what}"),
            (AboveCrown, depth) if entry.circular => {
                let below_crown = rise * (1.0 - f64::EPSILON / 2.0);
                assert!(
                    depth.is_some_and(|(lower, upper)| lower <= below_crown && upper == rise),
                    "{what}"
                );
            }
            (AboveCrown, depth) => assert_eq!(depth, None, "{what}"),
            (_, depth) => assert!(depth.is_some(), "{what}"),
        }
    }
    Ok(())
}

#[test]
fn zero_and_tiny_discharges_stay_at_the_floor_without_error() -> TestResult {
    let barrels = vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::rectangular_box(2.0, 1.0)?,
        CulvertBarrel::circular(3.7)?,
        CulvertBarrel::rectangular_box(1.0, 2f64.powi(-900))?,
        CulvertBarrel::circular(1.0)?,
    ];
    let discharges = [0.0, 1e-300, 1e-300, 0.0, f64::from_bits(1)];
    let rows = critical(barrels.clone(), &discharges, 9.81)?;
    for ((barrel, row), discharge) in barrels.iter().zip(&rows).zip(discharges) {
        let floor = barrel.rise_m() * 2f64.powi(-30);
        if discharge == 0.0 {
            assert_eq!(row.status, CriticalDepthStatus::Dry);
            assert_eq!(row.depth_m, Some((0.0, 0.0)));
        } else {
            assert_eq!(row.status, CriticalDepthStatus::BelowFloor, "{row:?}");
            let (lower, upper) = row.depth_m.unwrap_or((f64::NAN, f64::NAN));
            assert!(
                lower == 0.0 && floor <= upper && upper <= 2.0 * floor,
                "{row:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn scaled_barrels_solve_at_the_same_depth_fraction() -> TestResult {
    // Lengths scale by 2^-400 and Q by 2^-1000 = (2^-400)^(5/2): the critical fraction is unchanged.
    let scale = 2f64.powi(-400);
    let barrels = vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::circular(scale)?,
        CulvertBarrel::rectangular_box(2.5, 0.8)?,
        CulvertBarrel::rectangular_box(2.5 * scale, 0.8 * scale)?,
    ];
    let flow = 2f64.powi(-1000);
    let discharges = [0.5, 0.5 * flow, 1.0, flow];
    let rows = critical(barrels, &discharges, 9.81)?;
    for row in &rows {
        assert_eq!(row.status, CriticalDepthStatus::Solved, "{row:?}");
    }
    let fraction = |row: &Row, rise: f64| row.depth_m.map(|(l, u)| (l / rise, u / rise));
    for (plain, scaled, rise) in [(0, 1, 1.0), (2, 3, 0.8)] {
        let (a, b) = (
            fraction(&rows[plain], rise),
            fraction(&rows[scaled], rise * scale),
        );
        assert!(
            matches!((a, b), (Some(a), Some(b)) if a.0 <= b.1 && b.0 <= a.1),
            "{a:?} and {b:?} are disjoint"
        );
    }
    Ok(())
}

#[test]
fn tiny_circles_report_the_crown_without_underflow() -> TestResult {
    // Below a rise near 2^-430 every represented discharge exceeds the free-surface range.
    // The residual reads unit geometry, so the metre area never underflows to zero.
    let rise = 2f64.powi(-900);
    let barrels = vec![
        CulvertBarrel::circular(rise)?,
        CulvertBarrel::rectangular_box(1.0, rise)?,
    ];
    let rows = critical(barrels, &[1e-300, 1e-300], 9.81)?;
    assert_eq!(rows[0].status, CriticalDepthStatus::AboveCrown);
    assert!(
        rows[0]
            .depth_m
            .is_some_and(|(lower, upper)| 0.0 < lower && lower < upper && upper == rise)
    );
    assert_eq!(rows[1].status, CriticalDepthStatus::AboveCrown);
    assert_eq!(rows[1].depth_m, None);
    Ok(())
}

#[test]
fn box_capacity_separates_solved_crown_and_above_crown() -> TestResult {
    let (span, rise, gravity) = (2.5_f64, 0.8_f64, 9.81_f64);
    let capacity = gravity.sqrt() * span * rise.powf(1.5);
    let factors = [0.5, 0.999, 1.0, 1.001, 2.0];
    let barrels = vec![CulvertBarrel::rectangular_box(span, rise)?; factors.len()];
    let discharges: Vec<f64> = factors.iter().map(|f| f * capacity).collect();
    let rows = critical(barrels, &discharges, gravity)?;
    assert_eq!(rows[0].status, CriticalDepthStatus::Solved);
    assert_eq!(rows[1].status, CriticalDepthStatus::Solved);
    assert!(matches!(
        rows[2].status,
        CriticalDepthStatus::Solved
            | CriticalDepthStatus::AtCrown
            | CriticalDepthStatus::AboveCrown
    ));
    for row in &rows[3..] {
        assert_eq!(row.status, CriticalDepthStatus::AboveCrown);
        assert_eq!(row.depth_m, None);
    }
    Ok(())
}

#[test]
fn circle_near_crown_keeps_a_root_below_the_crown() -> TestResult {
    let gravity = 9.81;
    let discharges = [48.0, 1e3, 1e5, 1e8];
    let barrels = vec![CulvertBarrel::circular(1.0)?; discharges.len()];
    let rows = critical(barrels, &discharges, gravity)?;
    for row in &rows {
        let (lower, upper) = row.depth_m.unwrap_or((f64::NAN, f64::NAN));
        assert!(lower > 0.99 && upper <= 1.0, "{row:?}");
    }
    assert_eq!(rows[0].status, CriticalDepthStatus::Solved);
    assert_eq!(rows[3].status, CriticalDepthStatus::AboveCrown);
    Ok(())
}

#[test]
fn critical_depth_rises_with_discharge_without_certified_reversal() -> TestResult {
    let rank = |status: CriticalDepthStatus| match status {
        CriticalDepthStatus::Dry => 0,
        CriticalDepthStatus::BelowFloor => 1,
        CriticalDepthStatus::Solved => 2,
        CriticalDepthStatus::AtCrown => 3,
        CriticalDepthStatus::AboveCrown => 4,
    };
    let shapes = [
        (CulvertBarrel::circular(1.0)?, -30.0, 9.0),
        (CulvertBarrel::rectangular_box(2.5, 0.8)?, -22.0, 1.5),
    ];
    for (shape, low, high) in shapes {
        let discharges: Vec<f64> = (0..65)
            .map(|i| 10f64.powf(low + (high - low) * f64::from(i) / 64.0))
            .collect();
        let rows = critical(vec![shape; discharges.len()], &discharges, 9.81)?;
        let mut seen = std::collections::BTreeSet::new();
        for (pair, q) in rows.windows(2).zip(&discharges) {
            assert!(
                rank(pair[0].status) <= rank(pair[1].status),
                "{shape:?} at Q={q}"
            );
            if let (Some(before), Some(after)) = (pair[0].depth_m, pair[1].depth_m) {
                assert!(after.1 >= before.0, "{shape:?} reverses after Q={q}");
            }
            seen.insert(rank(pair[0].status));
        }
        // The sweep spans the floor, the solved range and the crown.
        assert!(
            seen.contains(&1) && seen.contains(&2),
            "{shape:?}: {seen:?}"
        );
        assert_eq!(rows.last().map(|r| rank(r.status)), Some(4), "{shape:?}");
    }
    Ok(())
}
