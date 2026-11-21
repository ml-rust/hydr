//! Manning normal depth against the frozen independent roots, with typed outcomes,
//! capacity limits, scale invariance and monotonicity.

use super::normal_rows::{NORMAL, NORMAL_SHA256, NormalClass};
use super::reference_rows::{BARRELS, REFERENCE_SHA256};
use super::support::{TestResult, barrel, batch, tensor};
use hydr::structures::{CulvertBarrel, CulvertBarrelDepthRow, NormalDepthStatus};
use std::f64::consts::PI;

type Row = CulvertBarrelDepthRow<NormalDepthStatus>;

/// Manning roughness of every row unless a test varies it.
const ROUGHNESS: f64 = 0.013;

/// Unit-circle depth fraction of the conveyance peak, rounded up.
const PEAK_FRACTION: f64 = 0.938_181_216_160_607_1;

/// Peak conveyance `A^(5/3) / P^(2/3)` of a unit circle.
const PEAK_CONVEYANCE: f64 = 0.335_281_968_047_520_6;

fn normal(
    barrels: Vec<CulvertBarrel>,
    discharges: &[f64],
    roughness: &[f64],
    slopes: &[f64],
) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    Ok(batch(barrels)?
        .normal_depth(&tensor(discharges)?, &tensor(roughness)?, &tensor(slopes)?)?
        .project()?)
}

/// Normal depths of `barrels` at one roughness and one slope.
fn uniform(
    barrels: Vec<CulvertBarrel>,
    discharges: &[f64],
    slope: f64,
) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    let m = discharges.len();
    normal(barrels, discharges, &vec![ROUGHNESS; m], &vec![slope; m])
}

fn holds(depth: Option<(f64, f64)>, root: [u64; 2]) -> bool {
    depth.is_some_and(|(lower, upper)| {
        lower <= f64::from_bits(root[0]) && f64::from_bits(root[1]) <= upper
    })
}

/// Full-barrel conveyance `A^(5/3) / P^(2/3)` of a circle of diameter `rise`.
fn circle_full_conveyance(rise: f64) -> f64 {
    (PI / 4.0).powf(5.0 / 3.0) / PI.powf(2.0 / 3.0) * rise.powf(8.0 / 3.0)
}

/// Open-section conveyance of a box just below its crown.
fn box_crown_conveyance(span: f64, rise: f64) -> f64 {
    (span * rise).powf(5.0 / 3.0) / (span + 2.0 * rise).powf(2.0 / 3.0)
}

/// Discharge `K sqrt(S) / n` of conveyance `K`.
fn manning(conveyance: f64, slope: f64) -> f64 {
    conveyance * slope.sqrt() / ROUGHNESS
}

fn rank(status: NormalDepthStatus) -> u8 {
    match status {
        NormalDepthStatus::NonPositiveSlope => 0,
        NormalDepthStatus::Dry => 1,
        NormalDepthStatus::BelowFloor => 2,
        NormalDepthStatus::Solved => 3,
        NormalDepthStatus::NearCapacity => 4,
        NormalDepthStatus::AboveCapacity => 5,
    }
}

/// Asserts that ranks never fall and enclosures never certifiably reverse along `rows`.
fn assert_rising(rows: &[Row], what: &str) {
    for (step, pair) in rows.windows(2).enumerate() {
        assert!(
            rank(pair[0].status) <= rank(pair[1].status),
            "{what} at step {step}: {pair:?}"
        );
        if let (Some(before), Some(after)) = (pair[0].depth_m, pair[1].depth_m) {
            assert!(after.1 >= before.0, "{what} reverses at step {step}");
        }
    }
}

#[test]
fn normal_fixture_is_the_frozen_file() {
    assert_eq!(NORMAL_SHA256, REFERENCE_SHA256);
    assert_eq!(NORMAL.len(), 3 * (3 * 11 + 2) + 3 * (3 * 10 + 2));
}

#[test]
fn reference_roots_lie_inside_enclosures_with_matching_outcomes() -> TestResult {
    use NormalDepthStatus::{
        AboveCapacity, BelowFloor, Dry, NearCapacity, NonPositiveSlope, Solved,
    };
    // One row per reference root: circles and boxes interleave in one batch.
    let barrels = NORMAL
        .iter()
        .map(|row| barrel(&BARRELS[row.barrel]))
        .collect::<Result<Vec<_>, _>>()?;
    let bits = |pick: fn(&super::normal_rows::NormalRow) -> u64| -> Vec<f64> {
        NORMAL.iter().map(|row| f64::from_bits(pick(row))).collect()
    };
    let rows = normal(
        barrels,
        &bits(|row| row.discharge),
        &bits(|row| row.manning_n),
        &bits(|row| row.slope),
    )?;
    for (reference, row) in NORMAL.iter().zip(&rows) {
        let entry = &BARRELS[reference.barrel];
        let rise = f64::from_bits(entry.rise);
        let slope = f64::from_bits(reference.slope);
        let what = format!("{} at {} S={slope}: {row:?}", entry.name, reference.label);
        let admitted: &[NormalDepthStatus] = match reference.class {
            NormalClass::Dry => &[Dry],
            NormalClass::NonPositiveSlope => &[NonPositiveSlope],
            NormalClass::BelowFloor => &[BelowFloor],
            NormalClass::NearFloor => &[BelowFloor, Solved],
            NormalClass::Free => &[Solved],
            NormalClass::NearTop => &[Solved, NearCapacity],
            NormalClass::NearCapacity => &[NearCapacity],
            NormalClass::NearCapacityAbsent => &[NearCapacity, AboveCapacity],
            NormalClass::AboveCapacity => &[AboveCapacity],
        };
        assert!(admitted.contains(&row.status), "{what}");
        if let Some(root) = reference.root {
            assert!(holds(row.depth_m, root), "{what} misses {root:x?}");
        }
        match (row.status, row.depth_m) {
            (Dry, depth) => assert_eq!(depth, Some((0.0, 0.0)), "{what}"),
            (BelowFloor, Some((lower, upper))) => {
                // The circle slope floor 5.42e-5 holds up to s_N, so a residual that straddles
                // zero at the floor reaches further than the steep local slope there.
                assert_eq!(lower, 0.0, "{what}");
                assert!(upper <= rise * 2f64.powi(-26), "{what}");
            }
            (Solved, Some((lower, upper))) if reference.class == NormalClass::Free => {
                let scale = f64::from_bits(reference.root.map_or(0, |r| r[1]));
                let allowance = reference.relative_width * scale;
                assert!(
                    upper - lower <= allowance,
                    "{what} wider than {allowance:e}"
                );
            }
            (NearCapacity, Some((_, upper))) if entry.circular => {
                assert!(
                    upper < rise && upper <= (rise * PEAK_FRACTION).next_up(),
                    "{what}"
                );
            }
            (NearCapacity, Some((_, upper))) => assert_eq!(upper, rise, "{what}"),
            (AboveCapacity | NonPositiveSlope, depth) => assert_eq!(depth, None, "{what}"),
            (_, depth) => assert!(depth.is_some(), "{what}"),
        }
    }
    Ok(())
}

#[test]
fn zero_tiny_and_unsloped_rows_take_their_typed_outcomes() -> TestResult {
    let circle = CulvertBarrel::circular(1.0)?;
    let rectangle = CulvertBarrel::rectangular_box(2.0, 1.0)?;
    let barrels = vec![
        circle, rectangle, circle, rectangle, circle, rectangle, circle,
    ];
    let discharges = [0.0, 0.0, 1e-300, 1e-300, 1.0, 1.0, 0.0];
    let slopes = [0.01, 0.01, 0.01, 0.01, 0.0, -0.02, -0.0];
    let rows = normal(barrels.clone(), &discharges, &[ROUGHNESS; 7], &slopes)?;
    for ((barrel, row), (discharge, slope)) in
        barrels.iter().zip(&rows).zip(discharges.iter().zip(slopes))
    {
        let floor = barrel.rise_m() * 2f64.powi(-30);
        if slope <= 0.0 {
            assert_eq!(row.status, NormalDepthStatus::NonPositiveSlope, "{row:?}");
            assert_eq!(row.depth_m, None);
        } else if *discharge == 0.0 {
            assert_eq!(row.status, NormalDepthStatus::Dry);
            assert_eq!(row.depth_m, Some((0.0, 0.0)));
        } else {
            assert_eq!(row.status, NormalDepthStatus::BelowFloor, "{row:?}");
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
fn circle_capacity_separates_rising_limb_peak_and_above() -> TestResult {
    // Full conveyance recurs at 0.8196 D. At s_N conveyance lies 1.1e-11 relatively below the
    // peak, so a target 3e-12 below the peak has its root between s_N and s_K.
    let slope = 0.01;
    let full = manning(circle_full_conveyance(1.0), slope);
    let peak = manning(PEAK_CONVEYANCE, slope);
    let discharges = [full, peak * (1.0 - 3e-12), peak * (1.0 + 1e-9), 2.0 * peak];
    let barrels = vec![CulvertBarrel::circular(1.0)?; discharges.len()];
    let rows = uniform(barrels, &discharges, slope)?;
    assert_eq!(rows[0].status, NormalDepthStatus::Solved);
    assert!(
        rows[0]
            .depth_m
            .is_some_and(|(lower, upper)| 0.8196 < lower && upper < 0.8197),
        "{:?}",
        rows[0]
    );
    assert_eq!(rows[1].status, NormalDepthStatus::NearCapacity);
    assert!(
        rows[1]
            .depth_m
            .is_some_and(|(lower, upper)| lower <= 0.93818 && 0.938181 < upper && upper < 1.0),
        "{:?}",
        rows[1]
    );
    for row in &rows[2..] {
        assert_eq!(row.status, NormalDepthStatus::AboveCapacity);
        assert_eq!(row.depth_m, None);
    }
    Ok(())
}

#[test]
fn box_rises_to_the_open_section_limit_at_the_crown() -> TestResult {
    // The open-section limit K(D^-) exceeds the full-barrel conveyance, which solves below D.
    let (span, rise, slope) = (2.5_f64, 0.8_f64, 0.01_f64);
    let crown = manning(box_crown_conveyance(span, rise), slope);
    let area = span * rise;
    let full = manning(
        area.powf(5.0 / 3.0) / (2.0 * (span + rise)).powf(2.0 / 3.0),
        slope,
    );
    let discharges = [full, 0.999 * crown, crown, 1.001 * crown, 2.0 * crown];
    let barrels = vec![CulvertBarrel::rectangular_box(span, rise)?; discharges.len()];
    let rows = uniform(barrels, &discharges, slope)?;
    for row in &rows[..2] {
        assert_eq!(row.status, NormalDepthStatus::Solved, "{row:?}");
        assert!(row.depth_m.is_some_and(|(_, upper)| upper < rise));
    }
    match (rows[2].status, rows[2].depth_m) {
        (NormalDepthStatus::Solved, Some((_, upper))) => assert!(upper <= rise),
        (NormalDepthStatus::NearCapacity, Some((lower, upper))) => {
            assert!(lower < rise && upper == rise, "{:?}", rows[2]);
        }
        (NormalDepthStatus::AboveCapacity, None) => {}
        _ => return Err(format!("box at its crown limit: {:?}", rows[2]).into()),
    }
    for row in &rows[3..] {
        assert_eq!(row.status, NormalDepthStatus::AboveCapacity);
        assert_eq!(row.depth_m, None);
    }
    Ok(())
}

#[test]
fn scaled_barrels_solve_at_the_same_depth_fraction() -> TestResult {
    // Lengths scale by 2^-300 and Q by 2^-800 = (2^-300)^(8/3): the normal fraction is unchanged.
    let scale = 2f64.powi(-300);
    let barrels = vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::circular(scale)?,
        CulvertBarrel::rectangular_box(2.5, 0.8)?,
        CulvertBarrel::rectangular_box(2.5 * scale, 0.8 * scale)?,
    ];
    let flow = 2f64.powi(-800);
    let discharges = [0.5, 0.5 * flow, 3.0, 3.0 * flow];
    let rows = uniform(barrels, &discharges, 0.01)?;
    for row in &rows {
        assert_eq!(row.status, NormalDepthStatus::Solved, "{row:?}");
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
fn tiny_barrels_report_capacity_without_underflow() -> TestResult {
    // A 2^-900 rise has conveyance near 2^-2400, far below every represented target.
    // The residual reads unit geometry and logs, so no area or conveyance underflows.
    let rise = 2f64.powi(-900);
    let barrels = vec![
        CulvertBarrel::circular(rise)?,
        CulvertBarrel::rectangular_box(1.0, rise)?,
    ];
    let rows = uniform(barrels, &[1e-300, 1e-300], 0.01)?;
    for row in &rows {
        assert_eq!(row.status, NormalDepthStatus::AboveCapacity, "{row:?}");
        assert_eq!(row.depth_m, None);
    }
    Ok(())
}

#[test]
fn normal_depth_rises_with_discharge_without_certified_reversal() -> TestResult {
    let shapes = [
        (CulvertBarrel::circular(1.0)?, -22.0, 1.0),
        (CulvertBarrel::rectangular_box(2.5, 0.8)?, -17.0, 2.0),
    ];
    for (shape, low, high) in shapes {
        let discharges: Vec<f64> = (0..65)
            .map(|i| 10f64.powf(low + (high - low) * f64::from(i) / 64.0))
            .collect();
        let rows = uniform(vec![shape; discharges.len()], &discharges, 0.01)?;
        assert_rising(&rows, &format!("{shape:?}"));
        let seen: std::collections::BTreeSet<u8> = rows.iter().map(|r| rank(r.status)).collect();
        // The sweep spans the floor, the solved range and the capacity limit.
        assert!(
            seen.contains(&2) && seen.contains(&3) && seen.contains(&5),
            "{shape:?}: {seen:?}"
        );
    }
    Ok(())
}

#[test]
fn normal_depth_rises_with_roughness_and_as_slope_falls() -> TestResult {
    for shape in [
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::rectangular_box(2.5, 0.8)?,
    ] {
        // 33 logarithmic steps from 10^low to 10^high.
        let ramp = |low: f64, high: f64| -> Vec<f64> {
            (0..33)
                .map(|i| 10f64.powf(low + (high - low) * f64::from(i) / 32.0))
                .collect()
        };
        let steps = 33;
        let barrels = vec![shape; steps];
        let discharge = vec![1.0; steps];
        let roughness = ramp(-3.0, -0.5);
        let rows = normal(barrels.clone(), &discharge, &roughness, &vec![0.01; steps])?;
        assert_rising(&rows, &format!("{shape:?} as n rises"));
        let slopes = ramp(0.0, -8.0);
        let rows = normal(barrels, &discharge, &vec![ROUGHNESS; steps], &slopes)?;
        assert_rising(&rows, &format!("{shape:?} as S falls"));
        assert_eq!(
            rows.last().map(|r| r.status),
            Some(NormalDepthStatus::AboveCapacity)
        );
    }
    Ok(())
}
