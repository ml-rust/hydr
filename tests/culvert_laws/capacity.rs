//! The culvert law on a mild circle near its conveyance peak capacity, where the outlet march
//! approaches a nearly double root of `S - Sf`.

use super::round_trip::{ROUND_TRIP_TOLERANCE, control_headwater, invert};
use super::support::{Culvert, TestResult, close};
use hydr::structures::{CulvertBarrel, CulvertInlet, StructureRegime};

/// Peak capacity of the barrel below, near the F64 value nearest it.
const PEAK: f64 = 1.875_563_757_518_865;

/// A 1.2 m circle with an HDS-5 code 1 inlet, 60 m long at slope 0.002.
fn circle() -> TestResult<Culvert> {
    let mut culvert = Culvert::both(
        CulvertBarrel::circular(1.2)?,
        [0.12, 0.0],
        CulvertInlet::hds5(1)?,
        0.5,
    );
    culvert.length_m = 60.0;
    culvert.end_b = None;
    Ok(culvert)
}

#[test]
fn outlet_headwater_through_the_peak_capacity_inverts_to_its_discharge() -> TestResult {
    let culvert = circle()?;
    let mut q: Vec<f64> = (0..=30).map(|k| 1.8 + 0.005 * f64::from(k)).collect();
    q.push(PEAK);
    for tailwater in [0.05, 0.5] {
        let lanes = vec![culvert; q.len()];
        let tw = vec![tailwater; q.len()];
        let hw = control_headwater(&lanes, &q, &tw)?;
        let total: Vec<f64> = hw.iter().map(|h| h[0]).collect();
        let found = invert(&lanes, &total, &tw)?;
        for (index, ((discharge, regime, bounds), expected)) in found.iter().zip(&q).enumerate() {
            let [_, inlet, outlet] = hw[index];
            assert!(outlet > inlet, "lane {index}: hw {:?}", hw[index]);
            assert_eq!(
                *regime,
                StructureRegime::CulvertOutletControl,
                "lane {index}"
            );
            assert!(
                close(*discharge, *expected, ROUND_TRIP_TOLERANCE),
                "TW={tailwater} lane {index}: Q={discharge:e} in {bounds:?}, expected {expected:e}"
            );
        }
    }
    Ok(())
}

#[test]
fn heads_across_the_peak_capacity_give_a_rising_outlet_discharge() -> TestResult {
    let culvert = circle()?;
    let heads: Vec<f64> = (0..=30).map(|k| 1.2 + 0.01 * f64::from(k)).collect();
    for tailwater in [0.05, 0.5] {
        let lanes = vec![culvert; heads.len()];
        let found = invert(&lanes, &heads, &vec![tailwater; heads.len()])?;
        for (index, pair) in found.windows(2).enumerate() {
            if let [(before, _, _), (after, regime, _)] = pair {
                assert_eq!(
                    *regime,
                    StructureRegime::CulvertOutletControl,
                    "TW={tailwater} head {index}"
                );
                assert!(after > before, "TW={tailwater} head {index}: {found:?}");
            }
        }
    }
    Ok(())
}
