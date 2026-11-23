//! Arc barrel critical and normal depth against the frozen independent roots.

use super::depth_rows::{CERTIFIED, CRITICAL, Class, DEPTHS_SHA256, GRAVITY, MANNING_N, NORMAL};
use super::geometry::{Outcome, tensor};
use super::runtime::{Runtime, client};
use super::{TestResult, barrel};
use hydr::structures::{
    CriticalDepthStatus, CulvertBarrel, CulvertBarrelBatch, CulvertBarrelDepthRow,
    NormalDepthStatus,
};

/// Batch with one row per fixture row, each the row's barrel.
fn batch(owners: impl Iterator<Item = usize>) -> Outcome<CulvertBarrelBatch<Runtime>> {
    let barrels = owners
        .map(|index| {
            let entry = CERTIFIED.get(index).ok_or("row names no barrel")?;
            barrel(&entry.barrel)
        })
        .collect::<Outcome<Vec<CulvertBarrel>>>()?;
    Ok(CulvertBarrelBatch::new(client(), barrels)?)
}

fn holds(depth: Option<(f64, f64)>, root: Option<[u64; 2]>) -> bool {
    match (depth, root) {
        (_, None) => true,
        (Some((lower, upper)), Some([down, up])) => {
            lower <= f64::from_bits(down) && f64::from_bits(up) <= upper
        }
        (None, Some(_)) => false,
    }
}

/// `true` when a solved enclosure is no wider than `relative_width` times its root.
fn narrow(depth: Option<(f64, f64)>, root: Option<[u64; 2]>, relative_width: f64) -> bool {
    let scale = root.map_or(0.0, |[_, up]| f64::from_bits(up));
    depth.is_some_and(|(lower, upper)| upper - lower <= relative_width * scale)
}

fn rise(index: usize) -> f64 {
    CERTIFIED
        .get(index)
        .and_then(|entry| entry.barrel.inputs.get(1))
        .map_or(f64::NAN, |bits| f64::from_bits(*bits))
}

#[test]
fn depth_fixture_is_the_frozen_file() {
    assert_eq!(
        DEPTHS_SHA256,
        "52858a2e2daa57c74f5cff3193e81bf6e1132fd547c6dae8a11cb7c7754e36d4"
    );
    assert!(CRITICAL.iter().any(|row| row.class == Class::AboveCrown));
    assert!(NORMAL.iter().any(|row| row.class == Class::AboveCapacity));
    assert!(NORMAL.iter().any(|row| row.class == Class::NearPeak));
    // No root lies within 2^-30 relative of a domain end, so every class has one status.
    let ambiguous = [Class::NearFloor, Class::NearCrown];
    assert!(CRITICAL.iter().all(|row| !ambiguous.contains(&row.class)));
    assert!(NORMAL.iter().all(|row| !ambiguous.contains(&row.class)));
    for entry in CERTIFIED {
        let name = entry.barrel.name;
        let [widest, _] = entry.widest.map(f64::from_bits);
        let [peak, peak_up] = entry.peak.map(f64::from_bits);
        let [log, log_up] = entry.log_peak.map(f64::from_bits);
        assert!(widest < peak && peak <= peak_up && log <= log_up, "{name}");
        // The proof gives 2 below the widest depth, so the true minimum is at least 2 there.
        assert!(f64::from_bits(entry.critical_minimum) > 2.0, "{name}");
        let fractions: Vec<f64> = entry
            .rising
            .iter()
            .map(|(s, _)| f64::from_bits(*s))
            .collect();
        assert!(
            fractions.windows(2).all(|w| w[0] <= w[1]) && fractions.iter().all(|s| *s < peak),
            "{name}"
        );
    }
}

#[test]
fn critical_roots_lie_inside_enclosures_with_matching_outcomes() -> TestResult {
    use CriticalDepthStatus::{AboveCrown, AtCrown, BelowFloor, Dry, Solved};
    let batch = batch(CRITICAL.iter().map(|row| row.barrel))?;
    let flows: Vec<f64> = CRITICAL
        .iter()
        .map(|row| f64::from_bits(row.discharge))
        .collect();
    let rows: Vec<CulvertBarrelDepthRow<CriticalDepthStatus>> = batch
        .critical_depth(&tensor(&flows)?, f64::from_bits(GRAVITY))?
        .project()?;
    let mut failures = Vec::new();
    for (reference, row) in CRITICAL.iter().zip(&rows) {
        let name = CERTIFIED
            .get(reference.barrel)
            .map_or("?", |c| c.barrel.name);
        let d = rise(reference.barrel);
        let admitted: &[CriticalDepthStatus] = match reference.class {
            Class::Dry => &[Dry],
            Class::BelowFloor => &[BelowFloor],
            Class::NearFloor => &[BelowFloor, Solved],
            Class::Free => &[Solved],
            Class::NearCrown => &[Solved, AtCrown],
            _ => &[AboveCrown],
        };
        let shaped = match (row.status, row.depth_m) {
            (Dry, depth) => depth == Some((0.0, 0.0)),
            (BelowFloor, Some((lower, upper))) => lower == 0.0 && upper <= 2.0 * d * 2f64.powi(-30),
            (Solved, depth) if reference.class == Class::Free => {
                narrow(depth, reference.root, reference.relative_width)
            }
            (AtCrown | AboveCrown, Some((_, upper))) => upper == d,
            (_, depth) => depth.is_some(),
        };
        if !(admitted.contains(&row.status) && holds(row.depth_m, reference.root) && shaped) {
            failures.push(format!(
                "{name} {}: {row:?} against {:?}",
                reference.label, reference.root
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn normal_roots_lie_inside_enclosures_with_matching_outcomes() -> TestResult {
    use NormalDepthStatus::{
        AboveCapacity, BelowFloor, Dry, NearCapacity, NonPositiveSlope, Solved,
    };
    let batch = batch(NORMAL.iter().map(|row| row.barrel))?;
    let column = |f: fn(&super::depth_rows::NormalRow) -> u64| -> Vec<f64> {
        NORMAL.iter().map(|row| f64::from_bits(f(row))).collect()
    };
    let flows = column(|row| row.discharge);
    let slopes = column(|row| row.slope);
    let roughness = vec![f64::from_bits(MANNING_N); NORMAL.len()];
    let rows = batch
        .normal_depth(&tensor(&flows)?, &tensor(&roughness)?, &tensor(&slopes)?)?
        .project()?;
    let mut failures = Vec::new();
    for (reference, row) in NORMAL.iter().zip(&rows) {
        let name = CERTIFIED
            .get(reference.barrel)
            .map_or("?", |c| c.barrel.name);
        let d = rise(reference.barrel);
        let admitted: &[NormalDepthStatus] = match reference.class {
            Class::Dry => &[Dry],
            Class::BelowFloor => &[BelowFloor],
            Class::NearFloor => &[BelowFloor, Solved],
            Class::Free => &[Solved],
            Class::NearPeak => &[Solved, NearCapacity],
            Class::AboveCapacity => &[AboveCapacity],
            _ => &[NonPositiveSlope],
        };
        let shaped = match (row.status, row.depth_m) {
            (Dry, depth) => depth == Some((0.0, 0.0)),
            (BelowFloor, Some((lower, upper))) => lower == 0.0 && upper <= 2.0 * d * 2f64.powi(-30),
            (Solved, depth) if reference.class == Class::Free => {
                narrow(depth, reference.root, reference.relative_width)
            }
            (NearCapacity, Some((lower, upper))) => 0.0 < lower && upper < d,
            (AboveCapacity | NonPositiveSlope, depth) => depth.is_none(),
            (_, depth) => depth.is_some(),
        };
        if !(admitted.contains(&row.status) && holds(row.depth_m, reference.root) && shaped) {
            failures.push(format!(
                "{name} {}: {row:?} against {:?}",
                reference.label, reference.root
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn near_peak_targets_reach_capacity_inside_the_peak_enclosure() -> TestResult {
    // A target just below the peak conveyance has its root between s_N and s_K.
    let rows: Vec<_> = NORMAL
        .iter()
        .filter(|row| row.class == Class::NearPeak)
        .collect();
    let batch = batch(rows.iter().map(|row| row.barrel))?;
    let flows: Vec<f64> = rows
        .iter()
        .map(|row| f64::from_bits(row.discharge))
        .collect();
    let slopes: Vec<f64> = rows.iter().map(|row| f64::from_bits(row.slope)).collect();
    let roughness = vec![f64::from_bits(MANNING_N); rows.len()];
    let found = batch
        .normal_depth(&tensor(&flows)?, &tensor(&roughness)?, &tensor(&slopes)?)?
        .project()?;
    for (reference, row) in rows.iter().zip(&found) {
        let entry = CERTIFIED
            .get(reference.barrel)
            .ok_or("row names no barrel")?;
        let d = rise(reference.barrel);
        let peak = f64::from_bits(entry.peak[1]) * d;
        if row.status == NormalDepthStatus::NearCapacity {
            let (_, upper) = row.depth_m.ok_or("near capacity without an enclosure")?;
            assert!(
                upper >= peak && upper - peak <= 1e-6 * d,
                "{}: {row:?}",
                entry.barrel.name
            );
        }
    }
    Ok(())
}

#[test]
fn arc_rows_hold_rising_limb_knots_under_the_independent_minima() -> TestResult {
    // Knots rise to s_N and their floors fall to kappa_n. Each floor lies at or below the true
    // minimum of d r_n / ds over [2^-30, s] for the reference knots s at or below it.
    let mut barrels = vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::rectangular_box(2.0, 1.0)?,
    ];
    for entry in CERTIFIED {
        barrels.push(barrel(&entry.barrel)?);
    }
    let found = CulvertBarrelBatch::<Runtime>::new(client(), barrels)?.depth_certificates()?;
    let mut failures = Vec::new();
    for row in found.iter().take(2) {
        if !row.limb.iter().all(|&knot| knot == (0.5, 1.0)) {
            failures.push(format!("circle or box row read knots {:?}", row.limb));
        }
    }
    for (entry, row) in CERTIFIED.iter().zip(found.iter().skip(2)) {
        let below = |s: f64| {
            entry
                .rising
                .iter()
                .filter(|(at, _)| f64::from_bits(*at) <= s)
                .map(|(_, m)| f64::from_bits(*m))
                .next_back()
        };
        let (s_n, kn) = (row.normal_top, row.normal_slope);
        let held = row.limb.len() == 8
            && row.limb.windows(2).all(|w| match w {
                [(s, k), (t, l)] => s <= t && k >= l,
                _ => false,
            })
            && row
                .limb
                .iter()
                .all(|&(s, k)| kn <= k && s <= s_n && below(s).is_none_or(|m| k <= m));
        if !held {
            failures.push(format!("{}: knots {:?}", entry.barrel.name, row.limb));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
