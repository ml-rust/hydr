//! Frozen culvert reference corpus: source hashes and structural invariants of each fixture.

// Later culvert units read every field.
#[allow(dead_code)]
#[path = "culverts/coefficient_rows.rs"]
mod coefficient_rows;

// Later culvert units read every field.
#[allow(dead_code)]
#[path = "culverts/inlet_rows.rs"]
mod inlet_rows;

// Later culvert units read every field.
#[allow(dead_code)]
#[path = "culverts/swmm_rows.rs"]
mod swmm_rows;

// Later culvert units read every field.
#[allow(dead_code)]
#[path = "culverts/outlet_rows.rs"]
mod outlet_rows;

// Later culvert units read every field.
#[allow(dead_code)]
#[path = "culverts/published_rows.rs"]
mod published_rows;

// The coordinator cases read every value.
#[allow(dead_code)]
#[path = "culverts/drain_rows.rs"]
mod drain_rows;

use coefficient_rows::{CODES, COEFFICIENT_SHA256, Shape, VALUE_MISMATCHES};
use inlet_rows::{ANCHORS, BARRELS, INLET_SHA256, RELATIVE_TOLERANCE, ROWS, Regime, SLOPES};
use outlet_rows::{FULL_ROWS, OUTLET_SHA256, Outlet, OutletRow, PROFILE_ROWS};
use published_rows::{Basis, PUBLISHED_ROWS, PUBLISHED_SHA256};
use swmm_rows::{Branch, SWMM_ROWS, SWMM_SHA256, SwmmClass};

/// Source JSON hashes the fixtures were transcribed from.
const SOURCES: [(&str, &str); 6] = [
    (
        drain_rows::DRAIN_SHA256,
        "56472c6ca558e41319f6fa867c04e82f70718e4d5747505fde7a9f90688cb968",
    ),
    (
        COEFFICIENT_SHA256,
        "70581710e64cdb92a67094908c8dd6aa168a9f1fb7db1ef2ccc4ee557c70caf8",
    ),
    (
        INLET_SHA256,
        "e206933b7541a95e060d8ab7059b0159bbed4c8a805ebfe15f0ef95c24070f9b",
    ),
    (
        SWMM_SHA256,
        "e206933b7541a95e060d8ab7059b0159bbed4c8a805ebfe15f0ef95c24070f9b",
    ),
    (
        OUTLET_SHA256,
        "a4094a5ada4d420d3e83ffc3226bf904b93832901db916292c9edafb3eab92ba",
    ),
    (
        PUBLISHED_SHA256,
        "cee1d4bc03bf0f8e9929a2de2d70483f9f4f64f7ccf64a087ee1bc88dd5263c1",
    ),
];

type TestResult = Result<(), String>;

fn bounds(pair: [u64; 2]) -> (f64, f64) {
    (f64::from_bits(pair[0]), f64::from_bits(pair[1]))
}

fn ordered(pair: [u64; 2], what: &str) -> TestResult {
    let (low, high) = bounds(pair);
    if low.is_finite() && high.is_finite() && low <= high {
        Ok(())
    } else {
        Err(format!("{what}: bounds {low:e} > {high:e} or not finite"))
    }
}

fn count<T>(rows: &[T], keep: impl Fn(&T) -> bool) -> usize {
    rows.iter().filter(|row| keep(row)).count()
}

#[test]
fn fixtures_carry_their_pinned_source_hashes() -> TestResult {
    for (fixture, pinned) in SOURCES {
        if fixture != pinned {
            return Err(format!(
                "fixture hash {fixture} differs from pinned {pinned}"
            ));
        }
    }
    Ok(())
}

#[test]
fn coefficient_table_covers_every_swmm_code() -> TestResult {
    let codes: Vec<u8> = CODES.iter().map(|row| row.code).collect();
    let expected: Vec<u8> = (1..=57).collect();
    if codes != expected {
        return Err(format!("codes {codes:?} are not 1..=57 in order"));
    }
    let mitered: Vec<u8> = CODES.iter().filter(|r| r.mitered).map(|r| r.code).collect();
    if mitered != [5, 37, 46] {
        return Err(format!("mitered codes {mitered:?}"));
    }
    let circles = count(CODES, |r| r.admitted && r.shape == Shape::Circular);
    let boxes = count(CODES, |r| r.admitted && r.shape == Shape::Box);
    let others = count(CODES, |r| r.admitted && r.shape == Shape::Other);
    if (circles, boxes, others) != (8, 18, 0) {
        return Err(format!(
            "admitted circles {circles}, boxes {boxes}, others {others}"
        ));
    }
    if let Some(row) = CODES
        .iter()
        .find(|r| !(r.form == 1 || r.form == 2) || r.k <= 0.0)
    {
        return Err(format!(
            "code {} has form {} and K {}",
            row.code, row.form, row.k
        ));
    }
    let mismatches = [VALUE_MISMATCHES];
    if mismatches != [0] {
        return Err(format!("{mismatches:?} HDS-5 and SWMM value mismatches"));
    }
    Ok(())
}

#[test]
fn inlet_headwater_rises_with_discharge_on_every_curve() -> TestResult {
    let admitted = count(CODES, |r| r.admitted);
    let per_curve = ROWS.len() / (admitted * SLOPES.len());
    if per_curve * admitted * SLOPES.len() != ROWS.len() || per_curve != 20 {
        return Err(format!(
            "{} rows do not split into 20 per curve",
            ROWS.len()
        ));
    }
    if ANCHORS.len() != admitted * SLOPES.len() {
        return Err(format!("{} anchors for {admitted} codes", ANCHORS.len()));
    }
    for curve in ROWS.chunks(per_curve) {
        let first = curve.first().ok_or("empty curve")?;
        let what = format!("code {} slope {}", first.code, first.slope);
        let shape = BARRELS
            .get(first.barrel)
            .ok_or(format!("{what}: no barrel"))?;
        let entry = CODES
            .iter()
            .find(|r| r.code == first.code)
            .ok_or(format!("{what}: no coefficients"))?;
        if (entry.shape == Shape::Circular) != shape.0 {
            return Err(format!("{what}: barrel shape differs from the code"));
        }
        let mut regimes = Vec::new();
        for (before, after) in curve.iter().zip(curve.iter().skip(1)) {
            ordered(before.headwater, &what)?;
            let same_curve = (before.code, before.slope) == (after.code, after.slope);
            // Limit rows sit one ulp apart in discharge, so their headwater bounds can touch.
            let rising = f64::from_bits(before.discharge) < f64::from_bits(after.discharge)
                && bounds(before.headwater).0 <= bounds(after.headwater).1
                && bounds(before.headwater).1 <= bounds(after.headwater).1;
            if !same_curve || !rising {
                return Err(format!(
                    "{what}: {} -> {} does not rise",
                    before.label, after.label
                ));
            }
        }
        for row in curve {
            if regimes.last() != Some(&row.regime) {
                regimes.push(row.regime);
            }
            let critical = row.critical_depth.is_some();
            if critical != (row.regime == Regime::Unsubmerged && entry.form == 1) {
                return Err(format!("{what} {}: critical depth presence", row.label));
            }
        }
        if regimes != [Regime::Unsubmerged, Regime::Transition, Regime::Submerged] {
            return Err(format!("{what}: regimes {regimes:?}"));
        }
    }
    for anchor in ANCHORS {
        let (_, low) = bounds(anchor.at_unsubmerged_limit);
        let (high, _) = bounds(anchor.at_submerged_limit);
        if low >= high {
            return Err(format!(
                "code {} slope {}: transition falls",
                anchor.code, anchor.slope
            ));
        }
    }
    if !(RELATIVE_TOLERANCE > 0.0 && RELATIVE_TOLERANCE < 1e-9) {
        return Err(format!("inlet tolerance {RELATIVE_TOLERANCE}"));
    }
    Ok(())
}

/// SWMM's PI (`consts.h`), 10 digits.
#[expect(
    clippy::approx_constant,
    reason = "SWMM's circle geometry uses this 10-digit value of pi"
)]
const SWMM_PI: f64 = 3.141592654;

#[test]
fn swmm_rows_sit_within_their_tolerance_of_the_reference() -> TestResult {
    let non_mitered = count(CODES, |r| r.admitted && !r.mitered);
    let form1 = count(CODES, |r| r.admitted && !r.mitered && r.form == 1);
    if SWMM_ROWS.len() != non_mitered * SLOPES.len() * 8 {
        return Err(format!("{} SWMM rows", SWMM_ROWS.len()));
    }
    let stale = count(SWMM_ROWS, |r| r.class == SwmmClass::Form1StaleFlow);
    if stale != form1 * SLOPES.len() * 5 {
        return Err(format!("{stale} stale-flow rows for {form1} form-1 codes"));
    }
    // Relative error of SWMM's PI, times 2 for the headwater sensitivity to area, times 2 margin.
    let circle_tolerance = 4.0 * (SWMM_PI - std::f64::consts::PI) / std::f64::consts::PI;
    for row in SWMM_ROWS {
        let what = format!("SWMM code {} at {:?}", row.code, row.reference);
        ordered(row.reference, &what)?;
        let coefficients = CODES
            .iter()
            .find(|c| c.code == row.code)
            .ok_or(format!("{what}: unknown code"))?;
        if coefficients.mitered || !coefficients.admitted {
            return Err(format!("{what}: mitered or unadmitted code"));
        }
        let expected_class = if coefficients.form == 1 && row.branch == Branch::Unsubmerged {
            SwmmClass::Form1StaleFlow
        } else {
            SwmmClass::Agrees
        };
        if row.class != expected_class {
            return Err(format!(
                "{what}: class {:?}, rule gives {expected_class:?}",
                row.class
            ));
        }
        let expected_tolerance = if row.circular {
            circle_tolerance
        } else {
            1e-12
        };
        if (row.tolerance - expected_tolerance).abs() > expected_tolerance * 1e-4 {
            return Err(format!(
                "{what}: tolerance {:e}, expected {expected_tolerance:e}",
                row.tolerance
            ));
        }
        let (low, high) = bounds(row.reference);
        let swmm = f64::from_bits(row.swmm_headwater);
        let allowed = row.tolerance * high;
        match row.class {
            SwmmClass::Agrees => {
                let gap = (swmm - low).abs().max((swmm - high).abs());
                if row.replay_discharge.is_some() || gap > allowed {
                    return Err(format!("{what}: gap {gap:e} above tolerance {allowed:e}"));
                }
            }
            SwmmClass::Form1StaleFlow => {
                if row.replay_discharge != Some(row.discharge) {
                    return Err(format!(
                        "{what}: SWMM flow differs from the stale-flow replay"
                    ));
                }
                let gap = (swmm - low).abs().min((swmm - high).abs());
                if gap <= allowed {
                    return Err(format!(
                        "{what}: stale-flow gap {gap:e} not above {allowed:e}"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn outlet_row(row: &OutletRow) -> TestResult {
    let what = row.name;
    ordered(row.headwater, what)?;
    for pair in [
        row.inlet_head,
        row.critical_depth,
        row.normal_depth,
        row.outlet_depth,
    ]
    .into_iter()
    .flatten()
    {
        ordered(pair, what)?;
    }
    let length = f64::from_bits(row.inputs.length);
    let tailwater = f64::from_bits(row.inputs.tailwater);
    let (headwater, _) = bounds(row.headwater);
    match row.regime {
        Outlet::Full => {
            let rise = f64::from_bits(row.inputs.rise);
            if tailwater < rise || headwater <= rise {
                return Err(format!("{what}: full row below the crown"));
            }
        }
        Outlet::Jump => {
            let jump = row.jump.ok_or(format!("{what}: jump row without a jump"))?;
            let (position, _) = bounds(jump.position);
            let (_, upstream) = bounds(jump.upstream_depth);
            let (downstream, _) = bounds(jump.downstream_depth);
            let (critical, _) = bounds(row.critical_depth.ok_or("jump without yc")?);
            if !(position > 0.0
                && position < length
                && upstream < critical
                && critical < downstream)
            {
                return Err(format!(
                    "{what}: jump at {position} with {upstream} -> {downstream}"
                ));
            }
        }
        Outlet::Supercritical => {
            if row.outlet_depth.is_none() {
                return Err(format!("{what}: no outlet depth"));
            }
        }
        Outlet::CriticalWithFullExit => {
            let rise = f64::from_bits(row.inputs.rise);
            let crown = [rise.to_bits(); 2];
            if tailwater < rise
                || row.inlet_head != Some(crown)
                || row.critical_depth != Some(crown)
            {
                return Err(format!(
                    "{what}: critical row with a full outlet off the crown"
                ));
            }
        }
        Outlet::PartlyFull => {}
    }
    Ok(())
}

#[test]
fn outlet_rows_cover_each_regime() -> TestResult {
    for row in FULL_ROWS.iter().chain(PROFILE_ROWS) {
        outlet_row(row)?;
    }
    let regimes = [
        Outlet::Full,
        Outlet::PartlyFull,
        Outlet::Supercritical,
        Outlet::Jump,
        Outlet::CriticalWithFullExit,
    ]
    .map(|regime| count(PROFILE_ROWS, |r| r.regime == regime));
    if regimes != [2, 28, 6, 2, 1] || FULL_ROWS.len() != 18 {
        return Err(format!(
            "profile regimes {regimes:?}, {} full rows",
            FULL_ROWS.len()
        ));
    }
    if count(FULL_ROWS, |r| r.regime != Outlet::Full) != 0 {
        return Err("closed-form row outside full flow".into());
    }
    Ok(())
}

#[test]
fn published_rows_hold_pages_and_tolerances() -> TestResult {
    for row in PUBLISHED_ROWS {
        let what = format!("{} {:?}", row.example, row.quantity);
        if !(row.tolerance > 0.0 && row.published > 0.0 && (190..=290).contains(&row.pdf_page)) {
            return Err(format!(
                "{what}: tolerance {} page {}",
                row.tolerance, row.pdf_page
            ));
        }
        let within = (row.equation_value - row.published).abs() <= row.tolerance;
        if within != row.within {
            return Err(format!("{what}: within flag disagrees with its values"));
        }
        if row.design.is_some() == (row.basis == Basis::Equation) {
            return Err(format!("{what}: basis and inputs disagree"));
        }
    }
    let outside = count(PUBLISHED_ROWS, |r| !r.within);
    if PUBLISHED_ROWS.len() != 73 || outside != 4 {
        return Err(format!("{} rows, {outside} outside", PUBLISHED_ROWS.len()));
    }
    Ok(())
}
