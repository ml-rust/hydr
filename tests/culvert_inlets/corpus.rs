//! Every frozen inlet row and limit anchor against the headwater enclosures.

use super::coefficient_rows::{CODES, Shape};
use super::inlet_rows::{ANCHORS, RELATIVE_TOLERANCE, ROWS};
use super::lanes::reference_lanes;
use super::support::{TestResult, headwater, holds, regime_agrees};
use hydr::structures::{CulvertBarrelShape, CulvertInlet, CulvertInletForm};

#[test]
fn hds5_codes_equal_the_frozen_coefficient_table() -> TestResult {
    for entry in CODES {
        let inlet = CulvertInlet::hds5(entry.code);
        // Codes 30 to 47 admit for arc barrels, checked against this table by the arc corpus.
        if !entry.admitted && (30..=47).contains(&entry.code) {
            continue;
        }
        if !entry.admitted {
            if inlet.is_ok() {
                return Err(format!("code {} admits without a reference curve", entry.code).into());
            }
            continue;
        }
        let inlet = inlet?;
        let form = if entry.form == 1 {
            CulvertInletForm::One
        } else {
            CulvertInletForm::Two
        };
        let shape = match entry.shape {
            Shape::Circular => Some(CulvertBarrelShape::Circular),
            Shape::Box => Some(CulvertBarrelShape::Box),
            Shape::Other => None,
        };
        let expected = (
            form,
            [entry.k, entry.m, entry.c, entry.y],
            entry.mitered,
            shape,
        );
        let found = (
            inlet.form(),
            inlet.coefficients(),
            inlet.mitered(),
            inlet.barrel_shape(),
        );
        if found != expected {
            return Err(format!("code {}: {found:?}, table {expected:?}", entry.code).into());
        }
    }
    Ok(())
}

#[test]
fn every_inlet_row_lies_inside_its_enclosure_within_the_frozen_tolerance() -> TestResult {
    let lanes = reference_lanes()?;
    let found = headwater(&lanes)?;
    if found.len() != ROWS.len() {
        return Err(format!("{} rows for {} references", found.len(), ROWS.len()).into());
    }
    for (row, result) in ROWS.iter().zip(&found) {
        let what = format!("code {} slope {} at {}", row.code, row.slope, row.label);
        holds(result.headwater_m, row.headwater, RELATIVE_TOLERANCE, &what)?;
        if !regime_agrees(result.regime, row.regime) {
            return Err(format!(
                "{what}: regime {:?}, reference {:?}",
                result.regime, row.regime
            )
            .into());
        }
    }
    Ok(())
}

#[test]
fn limit_anchors_lie_within_tolerance_of_the_rows_beside_each_limit() -> TestResult {
    let lanes = reference_lanes()?;
    let found = headwater(&lanes)?;
    for anchor in ANCHORS {
        let curve = |label: &str| {
            ROWS.iter().zip(&found).find(|(row, _)| {
                (row.barrel, row.code, row.slope, row.label)
                    == (anchor.barrel, anchor.code, anchor.slope, label)
            })
        };
        let sides = [
            ("3.5-", anchor.at_unsubmerged_limit),
            ("3.5+", anchor.at_unsubmerged_limit),
            ("4-", anchor.at_submerged_limit),
            ("4+", anchor.at_submerged_limit),
        ];
        for (label, reference) in sides {
            let what = format!(
                "code {} slope {} anchor at {label}",
                anchor.code, anchor.slope
            );
            let (_, result) = curve(label).ok_or_else(|| format!("{what}: no row"))?;
            let (low, high) = (f64::from_bits(reference[0]), f64::from_bits(reference[1]));
            let reach = (high - result.headwater_m.0)
                .max(result.headwater_m.1 - low)
                .abs();
            if reach > RELATIVE_TOLERANCE * high.abs() {
                return Err(format!(
                    "{what}: enclosure {:?} lies {reach:e} from the anchor [{low:e}, {high:e}]",
                    result.headwater_m
                )
                .into());
            }
        }
    }
    Ok(())
}
