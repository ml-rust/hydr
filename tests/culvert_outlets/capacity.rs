//! A mild circle near its conveyance peak, where `S - Sf` nearly touches zero at `y_K`: a dense
//! discharge sweep through the peak capacity and the F64 discharges around it.

use super::lanes::Lane;
use super::outlet_rows::PROFILE_RELATIVE_TOLERANCE;
use super::support::{TestResult, evaluate, headwater};
use super::sweep::rising;
use hydr::structures::{
    CulvertBarrel, CulvertOutletHeadwaterRow, CulvertOutletRow, JumpStatus, OutletRegime,
};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// Peak capacity of the circle below at slope 0.002, near the F64 value nearest it.
const PEAK: f64 = 1.875_563_757_518_865;

/// Ulps on each side of `PEAK`.
const NEIGHBOURS: u64 = 64;

/// A 1.2 m circle, 60 m long at slope 0.002.
fn circle(discharge: f64, tailwater: f64) -> Outcome<Lane> {
    Ok(Lane {
        row: CulvertOutletRow {
            barrel: CulvertBarrel::circular(1.2)?,
            length_m: 60.0,
            manning_n: 0.013,
            entrance_loss: 0.5,
            slope: 0.002,
            gravity_m_per_s2: 9.81,
        },
        discharge,
        tailwater,
    })
}

/// Requires every row to be partly full with a headwater enclosure within the profile
/// tolerance of its value.
fn narrow(rows: &[CulvertOutletHeadwaterRow], what: &str) -> TestResult {
    for (index, row) in rows.iter().enumerate() {
        let (low, high) = row.headwater_m;
        if row.regime != OutletRegime::PartlyFull || high - low > PROFILE_RELATIVE_TOLERANCE * high
        {
            return Err(format!("{what}: point {index} is {row:?}").into());
        }
    }
    Ok(())
}

#[test]
fn headwater_resolves_and_rises_through_the_peak_capacity() -> TestResult {
    for tailwater in [0.05, 0.5] {
        let lanes = (0..=150)
            .map(|k| circle(1.8 + 0.001 * f64::from(k), tailwater))
            .collect::<Outcome<Vec<_>>>()?;
        let found = headwater(&lanes)?;
        let what = format!("sweep at TW={tailwater}");
        rising(&found, |_, _| false, &what)?;
        narrow(&found, &what)?;
    }
    Ok(())
}

#[test]
fn headwater_resolves_at_every_discharge_next_to_the_peak_capacity() -> TestResult {
    let bits = PEAK.to_bits();
    let lanes = (bits - NEIGHBOURS..=bits + NEIGHBOURS)
        .map(|bits| circle(f64::from_bits(bits), 0.05))
        .collect::<Outcome<Vec<_>>>()?;
    // The jump query runs the same march, and a mild barrel has no jump.
    let (found, jumps) = evaluate(&lanes)?;
    if let Some(jump) = jumps.iter().find(|row| row.status != JumpStatus::NotSteep) {
        return Err(format!("ulps around the peak: jump row {jump:?}").into());
    }
    rising(&found, |_, _| false, "ulps around the peak")?;
    narrow(&found, "ulps around the peak")
}
