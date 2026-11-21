//! HDS-5 worked-example inlet headwaters, converted from US units, against the enclosures.
//!
//! Appendix rows use an elliptical barrel and printed critical heads, and critical-depth rows
//! belong to the barrel tests, so only design headwater rows run here.

use super::lanes::Lane;
use super::published_rows::{Design, PUBLISHED_ROWS, Quantity};
use super::support::{TestResult, headwater};
use hydr::structures::{CulvertBarrel, CulvertInlet, CulvertInletRow};

const FOOT: f64 = 0.3048;

/// HDS-5's gravity of 32.2 ft/s2 in metres per second squared.
const US_GRAVITY: f64 = 32.2 * FOOT;

/// Relative gap allowed between the kernel and the 12-digit equation value.
const EQUATION_DIGITS: f64 = 1e-11;

fn lane(design: &Design) -> Result<Lane, Box<dyn std::error::Error>> {
    let rise = design.rise * FOOT;
    let barrel = match design.span {
        Some(span) if !design.circular => CulvertBarrel::rectangular_box(span * FOOT, rise)?,
        _ => CulvertBarrel::circular(rise)?,
    };
    Ok(Lane {
        row: CulvertInletRow {
            barrel,
            inlet: CulvertInlet::hds5(design.code)?,
            slope: design.slope,
            gravity_m_per_s2: US_GRAVITY,
        },
        discharge: design.discharge * FOOT * FOOT * FOOT,
    })
}

#[test]
fn design_headwaters_lie_within_their_published_tolerance() -> TestResult {
    let rows: Vec<_> = PUBLISHED_ROWS
        .iter()
        .filter(|row| row.within && matches!(row.quantity, Quantity::Headwater | Quantity::HwOverD))
        .filter_map(|row| row.design.map(|design| (row, design)))
        .collect();
    let lanes = rows
        .iter()
        .map(|(_, design)| lane(design))
        .collect::<Result<Vec<_>, _>>()?;
    let found = headwater(&lanes)?;
    if found.len() != rows.len() || rows.len() < 20 {
        return Err(format!("{} results for {} design rows", found.len(), rows.len()).into());
    }
    for ((row, design), result) in rows.iter().zip(&found) {
        let what = format!("{} {:?}", row.example, row.quantity);
        let divisor = match row.quantity {
            Quantity::HwOverD => design.rise * FOOT,
            _ => FOOT,
        };
        let (low, high) = (
            result.headwater_m.0 / divisor,
            result.headwater_m.1 / divisor,
        );
        let gap = (low - row.published)
            .abs()
            .max((high - row.published).abs());
        if gap > row.tolerance {
            return Err(format!(
                "{what}: [{low}, {high}] lies {gap} from {}, above {}",
                row.published, row.tolerance
            )
            .into());
        }
        let digits = (low - row.equation_value)
            .abs()
            .max((high - row.equation_value).abs());
        if digits > EQUATION_DIGITS * row.equation_value {
            return Err(format!(
                "{what}: [{low}, {high}] lies {digits:e} from the equation value {}",
                row.equation_value
            )
            .into());
        }
    }
    Ok(())
}
