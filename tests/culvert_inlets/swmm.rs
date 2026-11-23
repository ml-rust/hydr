//! SWMM 5.2.4 inlet rows, converted to SI, against the headwater enclosures.

use super::coefficient_rows::CODES;
use super::inlet_rows::RELATIVE_TOLERANCE;
use super::lanes::Lane;
use super::support::{TestResult, headwater, holds};
use super::swmm_rows::{SWMM_ROWS, SwmmClass, SwmmRow};
use hydr::structures::{CulvertBarrel, CulvertInlet, CulvertInletRow};

/// SWMM's gravity of 32.2 ft/s2 in metres per second squared.
const SWMM_GRAVITY: f64 = 32.2 * 0.3048;

fn lane(row: &SwmmRow) -> Result<Lane, Box<dyn std::error::Error>> {
    let rise = f64::from_bits(row.rise);
    let barrel = if row.circular {
        CulvertBarrel::circular(rise)?
    } else {
        CulvertBarrel::rectangular_box(f64::from_bits(row.span), rise)?
    };
    Ok(Lane {
        row: CulvertInletRow {
            barrel,
            inlet: CulvertInlet::hds5(row.code)?,
            slope: f64::from_bits(row.slope),
            gravity_m_per_s2: SWMM_GRAVITY,
            face_blockage: hydr::structures::BlockageFraction::NONE,
        },
        discharge: f64::from_bits(row.discharge),
    })
}

#[test]
fn swmm_rows_agree_within_their_fixed_tolerance_and_stale_form_one_rows_do_not() -> TestResult {
    let lanes = SWMM_ROWS.iter().map(lane).collect::<Result<Vec<_>, _>>()?;
    let found = headwater(&lanes)?;
    let (mut agrees, mut stale) = (0, 0);
    for (row, result) in SWMM_ROWS.iter().zip(&found) {
        let what = format!(
            "SWMM code {} slope {:#x} at {:#x}",
            row.code, row.slope, row.discharge
        );
        let mitered = CODES
            .iter()
            .any(|entry| entry.code == row.code && entry.mitered);
        if mitered {
            return Err(format!("{what}: a mitered code sits among the SWMM rows").into());
        }
        // The SI conversion of SWMM's inputs moves the exact headwater far less than this.
        holds(result.headwater_m, row.reference, RELATIVE_TOLERANCE, &what)?;
        let swmm = f64::from_bits(row.swmm_headwater);
        let (low, high) = result.headwater_m;
        let allowed = row.tolerance * high;
        match row.class {
            SwmmClass::Agrees => {
                let gap = (swmm - low).abs().max((swmm - high).abs());
                if gap > allowed {
                    return Err(format!("{what}: SWMM gap {gap:e} above {allowed:e}").into());
                }
                agrees += 1;
            }
            SwmmClass::Form1StaleFlow => {
                let gap = (swmm - low).abs().min((swmm - high).abs());
                if gap <= allowed {
                    return Err(format!("{what}: stale-flow gap {gap:e} within {allowed:e}").into());
                }
                stale += 1;
            }
        }
    }
    if agrees == 0 || stale == 0 {
        return Err(format!("{agrees} agreeing and {stale} stale rows").into());
    }
    Ok(())
}
