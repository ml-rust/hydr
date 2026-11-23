//! Inlet control above a floor and through a blocked face against the frozen independent rows.

use super::inlet_rows::{
    ANCHORS, FACE_FRACTIONS, GRAVITY, RELATIVE_TOLERANCE, ROWS, Regime, Row, SECTIONS, SLOPES,
};
use super::runtime::{Runtime, client};
use super::section_rows;
use super::sections::{Outcome, floored, tensor};
use hydr::structures::{
    BlockageFraction, CulvertBarrel, CulvertInlet, CulvertInletBatch, CulvertInletHeadwaterRow,
    CulvertInletRow, InletRegime,
};

/// The floored barrel of inlet section `index`.
fn barrel(index: usize) -> Outcome<CulvertBarrel> {
    let entry = SECTIONS.get(index).ok_or("row names no inlet section")?;
    let section = section_rows::SECTIONS
        .iter()
        .find(|s| s.name == entry.name && s.sediment_depth == entry.sediment_depth)
        .ok_or("inlet section names no evidence section")?;
    floored(section)
}

/// The inlet row of fixture row `row` with the face fraction `face`.
fn inlet_row(row: &Row, face: u64) -> Outcome<CulvertInletRow> {
    Ok(CulvertInletRow {
        barrel: barrel(row.section)?,
        inlet: CulvertInlet::hds5(u8::try_from(row.code)?)?,
        slope: f64::from_bits(*SLOPES.get(row.slope).ok_or("row names no slope")?),
        gravity_m_per_s2: f64::from_bits(GRAVITY),
        face_blockage: BlockageFraction::new(f64::from_bits(face))?,
    })
}

/// Headwater rows of one batch over `lanes`.
fn headwater(lanes: &[(CulvertInletRow, f64)]) -> Outcome<Vec<CulvertInletHeadwaterRow>> {
    let rows = lanes.iter().map(|lane| lane.0).collect();
    let batch = CulvertInletBatch::<Runtime>::new(client(), rows)?;
    let flows: Vec<f64> = lanes.iter().map(|lane| lane.1).collect();
    Ok(batch.headwater(&tensor(&flows)?)?.project()?)
}

fn label(row: &Row) -> String {
    let name = SECTIONS.get(row.section).map_or("?", |s| s.name);
    format!(
        "{name} code {} slope {} X {}",
        row.code, row.slope, row.label
    )
}

/// `true` when the band the kernel reports can hold the reference band.
fn band_matches(found: InletRegime, reference: Regime) -> bool {
    match reference {
        Regime::Unsubmerged => matches!(
            found,
            InletRegime::Unsubmerged | InletRegime::UnsubmergedLimit
        ),
        Regime::Transition => matches!(
            found,
            InletRegime::Transition | InletRegime::UnsubmergedLimit | InletRegime::SubmergedLimit
        ),
        Regime::Submerged => matches!(found, InletRegime::Submerged | InletRegime::SubmergedLimit),
    }
}

/// Requires `[lower, upper]` to hold `reference` within the frozen relative tolerance.
fn holds(lower: f64, upper: f64, reference: [u64; 2], what: &str) -> Outcome<()> {
    let (low, high) = (f64::from_bits(reference[0]), f64::from_bits(reference[1]));
    let allowed = RELATIVE_TOLERANCE * high;
    if !(lower <= low && high <= upper) {
        return Err(format!("{what}: [{lower:e}, {upper:e}] misses [{low:e}, {high:e}]").into());
    }
    if (high - lower).max(upper - low) > allowed {
        return Err(format!("{what}: [{lower:e}, {upper:e}] reaches past {allowed:e}").into());
    }
    Ok(())
}

#[test]
fn unblocked_rows_above_a_floor_lie_inside_their_enclosures() -> Outcome<()> {
    let lanes = ROWS
        .iter()
        .map(|row| Ok((inlet_row(row, 0)?, f64::from_bits(row.discharge))))
        .collect::<Outcome<Vec<_>>>()?;
    let found = headwater(&lanes)?;
    let mut errors = Vec::new();
    for (row, result) in ROWS.iter().zip(&found) {
        let what = label(row);
        if !band_matches(result.regime, row.regime) {
            errors.push(format!(
                "{what}: band {:?}, reference {:?}",
                result.regime, row.regime
            ));
            continue;
        }
        let (lower, upper) = result.headwater_m;
        if let Err(error) = holds(lower, upper, row.headwater, &what) {
            errors.push(error.to_string());
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n").into())
    }
}

/// The blocked discharge `(1 - f) Q_0` passes at the unblocked headwater. Its enclosure ends
/// bracket the exact discharge, and the headwater rises with discharge, so the hull of the
/// headwater at both ends holds the reference.
#[test]
fn blocked_rows_pass_the_open_share_of_the_unblocked_discharge() -> Outcome<()> {
    let mut lanes = Vec::new();
    let mut labels = Vec::new();
    for row in ROWS {
        for (face, blocked) in FACE_FRACTIONS.iter().zip(&row.blocked_discharge).skip(1) {
            for end in blocked {
                lanes.push((inlet_row(row, *face)?, f64::from_bits(*end)));
            }
            labels.push((row, *face));
        }
    }
    let found = headwater(&lanes)?;
    let mut errors = Vec::new();
    for (pair, (row, face)) in found.chunks(2).zip(&labels) {
        let [low, high] = pair else { continue };
        let what = format!("{} f={}", label(row), f64::from_bits(*face));
        if let Err(error) = holds(low.headwater_m.0, high.headwater_m.1, row.headwater, &what) {
            errors.push(error.to_string());
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n").into())
    }
}

#[test]
fn headwater_is_continuous_at_both_limits_above_a_floor() -> Outcome<()> {
    // The rows beside each limit are F64 neighbours of its discharge, and the anchors frozen at
    // the exact limits lie between their references.
    let mut lanes = Vec::new();
    let mut names = Vec::new();
    for anchor in ANCHORS {
        for (below, above, value) in [
            ("3.5-", "3.5+", anchor.hw_at_3_5),
            ("4-", "4+", anchor.hw_at_4),
        ] {
            for label in [below, above] {
                let row = ROWS
                    .iter()
                    .find(|row| {
                        row.section == anchor.section
                            && row.code == anchor.code
                            && row.slope == anchor.slope
                            && row.label == label
                    })
                    .ok_or("an anchor lacks the rows beside its limit")?;
                lanes.push((inlet_row(row, 0)?, f64::from_bits(row.discharge)));
            }
            names.push((anchor.section, anchor.code, below, value));
        }
    }
    let found = headwater(&lanes)?;
    for (pair, (section, code, limit, value)) in found.chunks(2).zip(&names) {
        let [low, high] = pair else { continue };
        let lower = low.headwater_m.0.min(high.headwater_m.0);
        let upper = low.headwater_m.1.max(high.headwater_m.1);
        let what = format!("section {section} code {code} at {limit}");
        holds(lower, upper, *value, &what)?;
        if upper - lower > 2.0 * RELATIVE_TOLERANCE * upper {
            return Err(format!("{what}: headwater jumps across the limit").into());
        }
    }
    Ok(())
}

#[test]
fn headwater_rises_with_the_face_blockage_and_keeps_its_bits_at_zero() -> Outcome<()> {
    let fractions = [0.0, 0.1, 0.25, 0.5, 0.9];
    let mut lanes = Vec::new();
    let mut plain = Vec::new();
    for row in ROWS.iter().step_by(7) {
        let base = inlet_row(row, 0)?;
        let flow = f64::from_bits(row.discharge);
        for f in fractions {
            lanes.push((
                CulvertInletRow {
                    face_blockage: BlockageFraction::new(f)?,
                    ..base
                },
                flow,
            ));
        }
        plain.push((base, flow));
    }
    let found = headwater(&lanes)?;
    let alone = headwater(&plain)?;
    for (chunk, unblocked) in found.chunks(fractions.len()).zip(&alone) {
        if chunk.first().map(|row| row.headwater_m) != Some(unblocked.headwater_m) {
            return Err(format!(
                "f = 0 beside blocked rows {:?}, alone {:?}",
                chunk.first(),
                unblocked
            )
            .into());
        }
        for pair in chunk.windows(2) {
            let [before, after] = pair else { continue };
            if after.headwater_m.0 <= before.headwater_m.1 {
                return Err(format!(
                    "headwater {:?} then {:?} as the face closes",
                    before.headwater_m, after.headwater_m
                )
                .into());
            }
        }
    }
    Ok(())
}

#[test]
fn inlet_layouts_of_width_1_33_and_65_return_single_row_bits() -> Outcome<()> {
    let mut lanes = vec![
        (
            CulvertInletRow {
                barrel: CulvertBarrel::circular(1.2)?,
                inlet: CulvertInlet::hds5(1)?,
                slope: 0.01,
                gravity_m_per_s2: 9.81,
                face_blockage: BlockageFraction::NONE,
            },
            1.3,
        ),
        (
            CulvertInletRow {
                barrel: CulvertBarrel::rectangular_box(2.0, 1.5)?,
                inlet: CulvertInlet::hds5(12)?,
                slope: 0.0,
                gravity_m_per_s2: 9.81,
                face_blockage: BlockageFraction::NONE,
            },
            4.0,
        ),
    ];
    for (k, row) in ROWS.iter().step_by(37).enumerate() {
        let face = FACE_FRACTIONS
            .get(k % FACE_FRACTIONS.len())
            .copied()
            .unwrap_or(0);
        lanes.push((inlet_row(row, face)?, f64::from_bits(row.discharge)));
    }
    let bits = |rows: &[CulvertInletHeadwaterRow]| -> Vec<(InletRegime, u64, u64)> {
        rows.iter()
            .map(|r| {
                (
                    r.regime,
                    r.headwater_m.0.to_bits(),
                    r.headwater_m.1.to_bits(),
                )
            })
            .collect()
    };
    let mut alone = Vec::new();
    for lane in &lanes {
        alone.extend(bits(&headwater(std::slice::from_ref(lane))?));
    }
    for width in [1, 33, 65] {
        let picked: Vec<_> = lanes.iter().cycle().take(width).copied().collect();
        let found = bits(&headwater(&picked)?);
        for (k, row) in found.iter().enumerate() {
            if Some(row) != alone.get(k % lanes.len()) {
                return Err(
                    format!("width {width} lane {k}: bits differ from its single row").into(),
                );
            }
        }
    }
    Ok(())
}
