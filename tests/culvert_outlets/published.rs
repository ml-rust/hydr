//! HDS-5 worked-example full-barrel losses, converted from US units, against the enclosures.
//!
//! Each design row runs with a tailwater high enough to keep the barrel full end to end, so
//! `H = HW_o - TW + S L` is the full-barrel loss of HDS-5 equation 3.5. HDS-5 prints these from
//! nomographs with `Ku = 29`, which rounds `2g / 1.486^2`; hydr uses `2g` in SI.

use super::lanes::Lane;
use super::published_rows::{PUBLISHED_ROWS, Quantity};
use super::support::{TestResult, headwater};
use hydr::structures::{CulvertBarrel, CulvertOutletRow, OutletRegime};

const FOOT: f64 = 0.3048;

/// HDS-5's gravity of 32.2 ft/s2 in metres per second squared.
const US_GRAVITY: f64 = 32.2 * FOOT;

#[test]
fn full_barrel_losses_lie_within_their_published_tolerance() -> TestResult {
    let mut lanes = Vec::new();
    let mut rows = Vec::new();
    for row in PUBLISHED_ROWS
        .iter()
        .filter(|row| row.within && row.quantity == Quantity::BarrelLoss)
    {
        let Some(design) = row.design else {
            continue;
        };
        let Some(manning) = design.manning else {
            continue;
        };
        let rise = design.rise * FOOT;
        let barrel = match design.span {
            Some(span) if !design.circular => CulvertBarrel::rectangular_box(span * FOOT, rise)?,
            _ => CulvertBarrel::circular(rise)?,
        };
        let length = design.length * FOOT;
        let fall = design.slope * length;
        lanes.push(Lane {
            row: CulvertOutletRow {
                barrel,
                length_m: length,
                manning_n: manning,
                entrance_loss: design.ke,
                slope: design.slope,
                gravity_m_per_s2: US_GRAVITY,
            },
            discharge: design.discharge * FOOT * FOOT * FOOT,
            tailwater: 2.0 * rise + fall.max(0.0),
        });
        rows.push(row);
    }
    if rows.len() < 8 {
        return Err(format!("{} barrel-loss rows", rows.len()).into());
    }
    let found = headwater(&lanes)?;
    for ((row, lane), result) in rows.iter().zip(&lanes).zip(&found) {
        if result.regime != OutletRegime::Full {
            return Err(format!("{}: regime {:?}", row.example, result.regime).into());
        }
        let drop = lane.row.slope * lane.row.length_m;
        let loss = |headwater: f64| (headwater - lane.tailwater + drop) / FOOT;
        let (low, high) = (loss(result.headwater_m.0), loss(result.headwater_m.1));
        let gap = (low - row.published)
            .abs()
            .max((high - row.published).abs());
        if gap > row.tolerance {
            return Err(format!(
                "{}: H [{low}, {high}] ft lies {gap} from {}, above {}",
                row.example, row.published, row.tolerance
            )
            .into());
        }
    }
    Ok(())
}
