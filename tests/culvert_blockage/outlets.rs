//! Outlet control above a floor and through a blocked entrance face against the frozen
//! independent rows.

use super::outlet_rows::{
    FACE_LOSS_RELATIVE_TOLERANCE, FACE_LOSSES, FULL_RELATIVE_TOLERANCE, Flow, GRAVITY,
    JUMP_POSITION_TOLERANCE, Outlet, PROFILE_RELATIVE_TOLERANCE, ROWS, SECTIONS,
};
use super::runtime::{Runtime, client};
use super::section_rows;
use super::sections::{Outcome, floored, tensor};
use hydr::structures::{
    BlockageFraction, CulvertBarrel, CulvertOutletBatch, CulvertOutletHeadwaterRow,
    CulvertOutletJumpRow, CulvertOutletRow, JumpStatus, OutletRegime,
};

/// One outlet query: its row and inputs.
#[derive(Clone, Copy, Debug)]
pub struct Lane {
    pub row: CulvertOutletRow,
    pub discharge: f64,
    pub tailwater: f64,
}

/// The floored barrel of outlet section `index`.
pub fn barrel(index: usize) -> Outcome<CulvertBarrel> {
    let entry = SECTIONS.get(index).ok_or("row names no outlet section")?;
    let section = section_rows::SECTIONS
        .iter()
        .find(|s| s.name == entry.name)
        .ok_or("outlet section names no evidence section")?;
    let barrel = floored(section)?;
    let floor = barrel.floor().ok_or("the section lost its floor")?;
    if floor.sediment_depth_m.to_bits() != entry.sediment_depth
        || floor.roughness_ratio.to_bits() != entry.roughness_ratio
    {
        return Err(format!("{}: floor differs from the outlet fixture", entry.name).into());
    }
    Ok(barrel)
}

/// The lane of one frozen row.
pub fn lane(row: &Outlet) -> Outcome<Lane> {
    Ok(Lane {
        row: CulvertOutletRow {
            barrel: barrel(row.section)?,
            length_m: f64::from_bits(row.length),
            manning_n: f64::from_bits(row.wall_manning),
            entrance_loss: f64::from_bits(row.ke),
            slope: f64::from_bits(row.slope),
            gravity_m_per_s2: f64::from_bits(GRAVITY),
            face_blockage: BlockageFraction::new(f64::from_bits(row.face_fraction))?,
        },
        discharge: f64::from_bits(row.discharge),
        tailwater: f64::from_bits(row.tailwater),
    })
}

fn batch(lanes: &[Lane]) -> Outcome<CulvertOutletBatch<Runtime>> {
    Ok(CulvertOutletBatch::<Runtime>::new(
        client(),
        lanes.iter().map(|lane| lane.row).collect(),
    )?)
}

fn inputs(lanes: &[Lane]) -> Outcome<[numr::tensor::Tensor<Runtime>; 2]> {
    let flows: Vec<f64> = lanes.iter().map(|lane| lane.discharge).collect();
    let tails: Vec<f64> = lanes.iter().map(|lane| lane.tailwater).collect();
    Ok([tensor(&flows)?, tensor(&tails)?])
}

/// Headwater rows of one batch over `lanes`.
pub fn headwater(lanes: &[Lane]) -> Outcome<Vec<CulvertOutletHeadwaterRow>> {
    let [discharge, tailwater] = inputs(lanes)?;
    Ok(batch(lanes)?.headwater(&discharge, &tailwater)?.project()?)
}

/// Headwater and jump rows of one batch over `lanes`.
pub fn jumps(
    lanes: &[Lane],
) -> Outcome<(Vec<CulvertOutletHeadwaterRow>, Vec<CulvertOutletJumpRow>)> {
    let [discharge, tailwater] = inputs(lanes)?;
    let found = batch(lanes)?.jump(&discharge, &tailwater)?;
    Ok((found.headwater().project()?, found.project()?))
}

fn label(row: &Outlet) -> String {
    let name = SECTIONS.get(row.section).map_or("?", |s| s.name);
    format!(
        "{name} {} f={}",
        row.name,
        f64::from_bits(row.face_fraction)
    )
}

/// Requires `bounds` to hold `reference` and reach at most `allowed` from it.
fn within(bounds: (f64, f64), reference: [u64; 2], allowed: f64, what: &str) -> Outcome<()> {
    let (low, high) = (f64::from_bits(reference[0]), f64::from_bits(reference[1]));
    if !(bounds.0 <= low && high <= bounds.1) {
        return Err(format!(
            "{what}: [{:e}, {:e}] misses the reference [{low:e}, {high:e}]",
            bounds.0, bounds.1
        )
        .into());
    }
    let reach = (high - bounds.0).max(bounds.1 - low);
    if reach > allowed {
        return Err(
            format!("{what}: reaches {reach:e} from the reference, above {allowed:e}").into(),
        );
    }
    Ok(())
}

/// [`within`] with a tolerance relative to the reference.
fn holds(bounds: (f64, f64), reference: [u64; 2], relative: f64, what: &str) -> Outcome<()> {
    let scale = f64::from_bits(reference[1]).abs();
    within(bounds, reference, relative * scale, what)
}

fn regime(flow: Flow) -> OutletRegime {
    match flow {
        Flow::Full => OutletRegime::Full,
        Flow::PartlyFull => OutletRegime::PartlyFull,
        Flow::Supercritical | Flow::Jump | Flow::CriticalFullOutlet => OutletRegime::InletCritical,
    }
}

/// Checks the jump fields of one row.
fn jump_holds(row: &Outlet, jump: &CulvertOutletJumpRow) -> Outcome<()> {
    let what = label(row);
    let tolerance = PROFILE_RELATIVE_TOLERANCE;
    match (row.flow, row.jump, row.outlet_depth) {
        (Flow::Jump, Some(expected), _) => {
            if jump.status != JumpStatus::InBarrel {
                return Err(format!("{what}: jump status {:?}", jump.status).into());
            }
            let allowed = JUMP_POSITION_TOLERANCE * f64::from_bits(row.length);
            within(
                jump.position_m,
                expected.position,
                allowed,
                &format!("{what} position"),
            )?;
            holds(
                jump.upstream_depth_m,
                expected.upstream_depth,
                tolerance,
                &format!("{what} upstream depth"),
            )?;
            holds(
                jump.downstream_depth_m,
                expected.downstream_depth,
                tolerance,
                &format!("{what} downstream depth"),
            )
        }
        (Flow::Supercritical, _, Some(outlet)) => {
            if jump.status != JumpStatus::Supercritical {
                return Err(format!("{what}: jump status {:?}", jump.status).into());
            }
            holds(
                jump.outlet_depth_m,
                outlet,
                tolerance,
                &format!("{what} outlet depth"),
            )
        }
        (Flow::CriticalFullOutlet, _, _) => match jump.status {
            JumpStatus::Unresolved => Ok(()),
            status => Err(format!("{what}: jump status {status:?}").into()),
        },
        (Flow::Full | Flow::PartlyFull, _, _) => match jump.status {
            JumpStatus::NotSteep => Ok(()),
            status => Err(format!("{what}: jump status {status:?}").into()),
        },
        _ => Err(format!("{what}: reference row lacks its jump fields").into()),
    }
}

/// Every frozen row with its tolerance: full rows at the closed-form tolerance, profile and
/// steep rows at the profile tolerance.
fn checked(rows: &[&Outlet]) -> Outcome<()> {
    let lanes = rows
        .iter()
        .map(|row| lane(row))
        .collect::<Outcome<Vec<_>>>()?;
    let (found, jumps) = jumps(&lanes)?;
    let mut errors = Vec::new();
    for ((row, result), jump) in rows.iter().zip(&found).zip(&jumps) {
        let what = label(row);
        if result.regime != regime(row.flow) {
            errors.push(format!(
                "{what}: regime {:?}, reference {:?}",
                result.regime, row.flow
            ));
            continue;
        }
        let tolerance = if row.name == "full" {
            FULL_RELATIVE_TOLERANCE
        } else {
            PROFILE_RELATIVE_TOLERANCE
        };
        let mut checks = vec![holds(
            result.headwater_m,
            row.headwater,
            tolerance,
            &format!("{what} headwater"),
        )];
        if let Some(inlet) = row.inlet_head {
            checks.push(holds(
                result.inlet_head_m,
                inlet,
                tolerance,
                &format!("{what} inlet head"),
            ));
        }
        if row.name != "full" {
            checks.push(jump_holds(row, jump));
        }
        errors.extend(
            checks
                .into_iter()
                .filter_map(Result::err)
                .map(|e| e.to_string()),
        );
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n").into())
    }
}

#[test]
fn full_rows_above_a_floor_lie_inside_their_enclosures() -> Outcome<()> {
    let rows: Vec<&Outlet> = ROWS.iter().filter(|row| row.name == "full").collect();
    assert_eq!(rows.len(), 72);
    checked(&rows)
}

#[test]
fn profile_rows_above_a_floor_lie_inside_their_enclosures() -> Outcome<()> {
    let rows: Vec<&Outlet> = ROWS
        .iter()
        .filter(|row| row.name != "full" && !row.name.starts_with("steep_"))
        .collect();
    assert_eq!(rows.len(), 72);
    checked(&rows)
}

#[test]
fn steep_rows_above_a_floor_lie_inside_their_enclosures() -> Outcome<()> {
    let rows: Vec<&Outlet> = ROWS
        .iter()
        .filter(|row| row.name.starts_with("steep_"))
        .collect();
    assert_eq!(rows.len(), 54);
    checked(&rows)
}

#[test]
fn face_entrance_loss_holds_the_frozen_values() -> Outcome<()> {
    // A full barrel isolates ke_f: HW = TW + (1 + ke_f + friction) V^2 / (2 g) - S L.
    let circle = CulvertBarrel::circular(1.2)?;
    let mut lanes = Vec::new();
    for loss in FACE_LOSSES {
        let row = CulvertOutletRow {
            barrel: circle,
            length_m: 1.0,
            manning_n: 0.012,
            entrance_loss: f64::from_bits(loss.ke),
            slope: 0.0,
            gravity_m_per_s2: 9.81,
            face_blockage: BlockageFraction::new(f64::from_bits(loss.face_fraction))?,
        };
        let unblocked = CulvertOutletRow {
            face_blockage: BlockageFraction::NONE,
            ..row
        };
        for row in [row, unblocked] {
            lanes.push(Lane {
                row,
                discharge: 2.0,
                tailwater: 2.4,
            });
        }
    }
    let rows = headwater(&lanes)?;
    for (loss, pair) in FACE_LOSSES.iter().zip(rows.chunks(2)) {
        let [blocked, open] = pair else { continue };
        let ke = f64::from_bits(loss.ke);
        // The headwater gap is (ke_f - ke) V^2 / (2 g), so ke_f follows from both rows.
        let velocity_head = (2.0 / (std::f64::consts::PI * 0.36)).powi(2) / (2.0 * 9.81);
        let gap = |a: f64, b: f64| (a - b) / velocity_head;
        let found = (
            ke + gap(blocked.headwater_m.0, open.headwater_m.1),
            ke + gap(blocked.headwater_m.1, open.headwater_m.0),
        );
        let expected = f64::from_bits(loss.face_ke[1]);
        let allowed = 1e-9 * expected + FACE_LOSS_RELATIVE_TOLERANCE;
        if !(found.0 - allowed <= expected && expected <= found.1 + allowed) {
            return Err(format!(
                "ke={ke} f={}: ke_f from the headwater gap {found:?}, frozen {expected}",
                f64::from_bits(loss.face_fraction)
            )
            .into());
        }
        if loss.face_fraction == 0 {
            assert_eq!(blocked.headwater_m, open.headwater_m);
        } else if blocked.headwater_m.0 <= open.headwater_m.1 {
            return Err(format!("ke={ke}: headwater does not rise with the face blockage").into());
        }
    }
    Ok(())
}
