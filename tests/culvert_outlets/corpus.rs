//! Every frozen full-barrel and profile row against the outlet enclosures.

use super::lanes::lane;
use super::outlet_rows::{
    FULL_RELATIVE_TOLERANCE, FULL_ROWS, JUMP_POSITION_TOLERANCE, Outlet, OutletRow,
    PROFILE_RELATIVE_TOLERANCE, PROFILE_ROWS,
};
use super::support::{TestResult, evaluate, headwater, holds, holds_within};
use hydr::structures::{JumpStatus, OutletRegime};

fn regime(row: &OutletRow) -> OutletRegime {
    match row.regime {
        Outlet::Full => OutletRegime::Full,
        Outlet::PartlyFull => OutletRegime::PartlyFull,
        Outlet::Supercritical | Outlet::Jump => OutletRegime::InletCritical,
    }
}

fn label(row: &OutletRow) -> String {
    let shape = if row.inputs.circular { "circle" } else { "box" };
    format!("{shape} {}", row.name)
}

#[test]
fn full_barrel_rows_lie_inside_their_enclosures_within_the_frozen_tolerance() -> TestResult {
    let lanes = FULL_ROWS.iter().map(lane).collect::<Result<Vec<_>, _>>()?;
    let found = headwater(&lanes)?;
    if found.len() != FULL_ROWS.len() || FULL_ROWS.len() != 18 {
        return Err(format!("{} rows for {} references", found.len(), FULL_ROWS.len()).into());
    }
    for (index, (row, result)) in FULL_ROWS.iter().zip(&found).enumerate() {
        let what = format!("full row {index}");
        if result.regime != OutletRegime::Full {
            return Err(format!("{what}: regime {:?}", result.regime).into());
        }
        holds(
            result.headwater_m,
            row.headwater,
            FULL_RELATIVE_TOLERANCE,
            &what,
        )?;
    }
    Ok(())
}

#[test]
fn profile_rows_lie_inside_their_enclosures_within_the_frozen_tolerance() -> TestResult {
    let lanes = PROFILE_ROWS
        .iter()
        .map(lane)
        .collect::<Result<Vec<_>, _>>()?;
    let found = headwater(&lanes)?;
    if found.len() != PROFILE_ROWS.len() || PROFILE_ROWS.len() != 20 {
        return Err(format!("{} rows for {} references", found.len(), PROFILE_ROWS.len()).into());
    }
    let tolerance = PROFILE_RELATIVE_TOLERANCE;
    for (row, result) in PROFILE_ROWS.iter().zip(&found) {
        let what = label(row);
        if result.regime != regime(row) {
            return Err(format!(
                "{what}: regime {:?}, reference {:?}",
                result.regime, row.regime
            )
            .into());
        }
        holds(
            result.headwater_m,
            row.headwater,
            tolerance,
            &format!("{what} headwater"),
        )?;
        if let Some(inlet) = row.inlet_head {
            holds(
                result.inlet_head_m,
                inlet,
                tolerance,
                &format!("{what} inlet head"),
            )?;
        }
    }
    Ok(())
}

#[test]
fn jump_rows_lie_inside_their_enclosures_within_the_frozen_tolerance() -> TestResult {
    let lanes = PROFILE_ROWS
        .iter()
        .map(lane)
        .collect::<Result<Vec<_>, _>>()?;
    let (_, jumps) = evaluate(&lanes)?;
    if jumps.len() != PROFILE_ROWS.len() {
        return Err(format!("{} rows for {} references", jumps.len(), PROFILE_ROWS.len()).into());
    }
    let tolerance = PROFILE_RELATIVE_TOLERANCE;
    for (row, jump) in PROFILE_ROWS.iter().zip(&jumps) {
        let what = label(row);
        let length = f64::from_bits(row.inputs.length);
        match (row.regime, row.jump, row.outlet_depth) {
            (Outlet::Jump, Some(expected), _) => {
                if jump.status != JumpStatus::InBarrel {
                    return Err(format!("{what}: jump status {:?}", jump.status).into());
                }
                let allowed = JUMP_POSITION_TOLERANCE * length;
                holds_within(
                    jump.position_m,
                    expected.position,
                    allowed,
                    &format!("{what} jump position"),
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
                )?;
            }
            (Outlet::Supercritical, _, Some(outlet)) => {
                if jump.status != JumpStatus::Supercritical {
                    return Err(format!("{what}: jump status {:?}", jump.status).into());
                }
                holds(
                    jump.outlet_depth_m,
                    outlet,
                    tolerance,
                    &format!("{what} outlet depth"),
                )?;
            }
            (Outlet::Full | Outlet::PartlyFull, _, _) => {
                if jump.status != JumpStatus::NotSteep {
                    return Err(format!("{what}: jump status {:?}", jump.status).into());
                }
            }
            _ => return Err(format!("{what}: reference row lacks its jump fields").into()),
        }
    }
    Ok(())
}
