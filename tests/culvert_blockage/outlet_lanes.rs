//! Outlet control above a floor: composite friction, continuity in the tailwater, monotone
//! face blockage, and lane independence beside rows without a floor.

use super::composite_rows::{FRICTION, HEAD_LOSS_RELATIVE_TOLERANCE};
use super::outlet_rows::{Outlet, PROFILE_RELATIVE_TOLERANCE, ROWS, SECTIONS};
use super::outlets::{Lane, barrel, headwater, jumps, lane};
use super::section_rows::{self, LAYOUT};
use super::sections::{Outcome, floored};
use hydr::structures::{
    BlockageFraction, CulvertAlignment, CulvertApplicability, CulvertBarrel, CulvertEnd,
    CulvertEnds, CulvertInlet, CulvertOutletHeadwaterRow, CulvertOutletJumpRow, CulvertOutletRow,
    CulvertParameters, JumpStatus, OutletRegime,
};

#[test]
fn culvert_parameters_reject_a_prefloored_barrel_naming_it() -> Outcome<()> {
    let barrel = barrel(0)?;
    let end = CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?;
    let result = CulvertParameters::new(
        barrel,
        CulvertAlignment {
            length_m: 40.0,
            invert_a_m: 0.04,
            invert_b_m: 0.0,
        },
        0.012,
        1,
        CulvertEnds {
            a: Some(end),
            b: None,
        },
        9.81,
        CulvertApplicability {
            coefficient_provenance: "HDS-5 Table A.1".into(),
            barrel_condition_assertion: "Sediment bed at 0.4 D".into(),
        },
    );
    let message = result.err().map(|e| e.to_string()).unwrap_or_default();
    if message.contains("CulvertParameters holds the unblocked barrel")
        && message.contains("sediment_depth_m=")
    {
        Ok(())
    } else {
        Err(format!("the law admitted a pre-floored barrel or named it wrongly: {message}").into())
    }
}

fn profile(section: usize, name: &str) -> Outcome<&'static Outlet> {
    ROWS.iter()
        .find(|row| row.section == section && row.name == name && row.face_fraction == 0)
        .ok_or_else(|| format!("no {name} row on section {section}").into())
}

#[test]
fn full_barrel_head_loss_is_composite_manning_friction() -> Outcome<()> {
    // No entrance loss and a level barrel: HW = TW + V^2 / (2 g) + n_c^2 V^2 L / R^(4/3).
    let mut lanes = Vec::new();
    for friction in FRICTION {
        let section = section_rows::SECTIONS
            .get(friction.section)
            .ok_or("friction row names no section")?;
        let culvert = floored(section)?;
        lanes.push(Lane {
            row: CulvertOutletRow {
                barrel: culvert,
                length_m: f64::from_bits(friction.length),
                manning_n: f64::from_bits(section.wall_manning),
                entrance_loss: 0.0,
                slope: 0.0,
                gravity_m_per_s2: 9.81,
                face_blockage: BlockageFraction::NONE,
            },
            discharge: f64::from_bits(friction.discharge),
            tailwater: 2.0 * culvert.open_rise_m(),
        });
    }
    let rows = headwater(&lanes)?;
    for ((friction, lane), row) in FRICTION.iter().zip(&lanes).zip(&rows) {
        let mid = |pair: [u64; 2]| 0.5 * (f64::from_bits(pair[0]) + f64::from_bits(pair[1]));
        let velocity = lane.discharge / mid(friction.full_area);
        let expected =
            lane.tailwater + velocity * velocity / (2.0 * 9.81) + mid(friction.head_loss);
        let allowed = HEAD_LOSS_RELATIVE_TOLERANCE * expected;
        let (low, high) = row.headwater_m;
        if row.regime != OutletRegime::Full || low - allowed > expected || expected > high + allowed
        {
            return Err(format!(
                "section {}: headwater {:?} {:?}, composite friction gives {expected}",
                friction.section, row.regime, row.headwater_m
            )
            .into());
        }
    }
    Ok(())
}

/// Depth of a named layout value of evidence section `name`, in metres.
fn junctions(name: &str) -> Outcome<Vec<f64>> {
    let index = section_rows::SECTIONS
        .iter()
        .position(|s| s.name == name)
        .ok_or("no such section")?;
    Ok(LAYOUT
        .iter()
        .filter(|v| v.section == index && v.name.starts_with("junction"))
        .map(|v| f64::from_bits(v.bounds[1]))
        .collect())
}

#[test]
fn headwater_is_continuous_across_critical_depth_junctions_and_the_crown() -> Outcome<()> {
    let mut lanes = Vec::new();
    let mut names = Vec::new();
    for (index, entry) in SECTIONS.iter().enumerate() {
        let mild = lane(profile(index, "m1")?)?;
        let critical = profile(index, "m2_critical_outlet")?
            .critical_depth
            .ok_or("m2 row lacks its critical depth")?;
        let rise = mild.row.barrel.open_rise_m();
        let mut seams = vec![
            ("critical depth".to_owned(), f64::from_bits(critical[1])),
            ("crown".to_owned(), rise),
        ];
        for junction in junctions(entry.name)? {
            if junction < rise {
                seams.push(("junction".to_owned(), junction));
            }
        }
        for (name, depth) in seams {
            for tailwater in [depth.next_down().next_down(), depth.next_up()] {
                lanes.push(Lane { tailwater, ..mild });
            }
            names.push(format!("{} {name}", entry.name));
        }
    }
    let rows = headwater(&lanes)?;
    for (pair, name) in rows.chunks(2).zip(&names) {
        let [below, above] = pair else { continue };
        if [below, above]
            .iter()
            .any(|row| row.regime == OutletRegime::Unresolved)
        {
            return Err(format!("{name}: unresolved").into());
        }
        let low = below.headwater_m.0.min(above.headwater_m.0);
        let high = below.headwater_m.1.max(above.headwater_m.1);
        let allowed = 2.0 * PROFILE_RELATIVE_TOLERANCE * high.abs();
        if high - low > allowed {
            return Err(format!(
                "{name}: headwater {:?} below and {:?} above differ by more than {allowed:e}",
                below.headwater_m, above.headwater_m
            )
            .into());
        }
    }
    Ok(())
}

#[test]
fn headwater_rises_with_the_face_blockage() -> Outcome<()> {
    let fractions = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9];
    let mut lanes = Vec::new();
    for index in 0..SECTIONS.len() {
        for name in [
            "full_mild",
            "m1",
            "m2_critical_outlet",
            "steep_supercritical",
        ] {
            let base = lane(profile(index, name)?)?;
            for f in fractions {
                let row = CulvertOutletRow {
                    face_blockage: BlockageFraction::new(f)?,
                    ..base.row
                };
                lanes.push(Lane { row, ..base });
            }
        }
    }
    let rows = headwater(&lanes)?;
    for (k, chunk) in rows.chunks(fractions.len()).enumerate() {
        for pair in chunk.windows(2) {
            let [before, after] = pair else { continue };
            if before.regime == OutletRegime::Unresolved
                || after.headwater_m.0 <= before.headwater_m.1
            {
                return Err(format!(
                    "sweep {k}: headwater {:?} then {:?} as the face closes",
                    before.headwater_m, after.headwater_m
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Lanes of floored barrels from the frozen rows, beside rows without a floor.
fn mixed() -> Outcome<Vec<Lane>> {
    let mut lanes = Vec::new();
    let plain = [
        CulvertBarrel::circular(1.2)?,
        CulvertBarrel::rectangular_box(2.0, 1.5)?,
        barrel(4)?.base(),
        barrel(6)?.base(),
    ];
    for (index, culvert) in plain.into_iter().enumerate() {
        let source = lane(profile(index + 3, "m1")?)?;
        lanes.push(Lane {
            row: CulvertOutletRow {
                barrel: culvert,
                ..source.row
            },
            discharge: source.discharge,
            tailwater: 0.6 * culvert.rise_m(),
        });
    }
    for (index, name) in [
        (0, "m1"),
        (1, "rising_to_crown"),
        (2, "full_mild"),
        (3, "steep_jump"),
        (4, "steep_supercritical"),
        (5, "m2_critical_outlet"),
        (6, "steep_jump"),
        (7, "steep_drowned"),
        (8, "m1"),
    ] {
        lanes.push(lane(profile(index, name)?)?);
    }
    let blocked = ROWS
        .iter()
        .find(|row| row.section == 6 && row.name == "m1" && row.face_fraction != 0)
        .ok_or("no blocked m1 row")?;
    lanes.push(lane(blocked)?);
    Ok(lanes)
}

type Bits = Vec<(OutletRegime, JumpStatus, [u64; 12])>;

fn bits(rows: &[CulvertOutletHeadwaterRow], jumps: &[CulvertOutletJumpRow]) -> Bits {
    rows.iter()
        .zip(jumps)
        .map(|(row, jump)| {
            let pairs = [
                row.headwater_m,
                row.inlet_head_m,
                jump.position_m,
                jump.upstream_depth_m,
                jump.downstream_depth_m,
                jump.outlet_depth_m,
            ];
            let mut values = [0; 12];
            for (slot, (a, b)) in values.chunks_mut(2).zip(pairs) {
                slot.copy_from_slice(&[a.to_bits(), b.to_bits()]);
            }
            (row.regime, jump.status, values)
        })
        .collect()
}

#[test]
fn outlet_layouts_of_width_1_33_and_65_return_single_row_bits() -> Outcome<()> {
    let lanes = mixed()?;
    let mut alone = Vec::new();
    for lane in &lanes {
        let (rows, found) = jumps(std::slice::from_ref(lane))?;
        alone.extend(bits(&rows, &found));
    }
    for width in [1, 33, 65] {
        let picked: Vec<Lane> = lanes.iter().cycle().take(width).copied().collect();
        let (rows, found) = jumps(&picked)?;
        for (k, found) in bits(&rows, &found).into_iter().enumerate() {
            if Some(&found) != alone.get(k % lanes.len()) {
                return Err(
                    format!("width {width} lane {k}: bits differ from its single row").into(),
                );
            }
        }
    }
    Ok(())
}
