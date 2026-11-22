//! The culvert law on tangent-arc barrels: round trip, control selection, equal heads,
//! reversal and dry ends.

use super::round_trip::{ROUND_TRIP_TOLERANCE, control_headwater, invert};
use super::support::{Culvert, TestResult, client, close, dry, elements, propose, wet};
use hydr::structures::{
    CulvertBarrel, CulvertInlet, FlowDirection, StructureBatch, StructureError,
    StructureInputErrorKind, StructureRegime, StructureStatus,
};
use numr::runtime::cpu::CpuRuntime;

/// Relative gap between the two control headwaters below which a lane may tie.
const SEPARATION: f64 = 1e-9;

/// One evidence barrel per arc shape with an HDS-5 code of its shape.
fn barrels() -> TestResult<[(CulvertBarrel, u8); 5]> {
    Ok([
        (
            CulvertBarrel::horizontal_ellipse(1.524, 0.9652, 0.419_475_92)?,
            30,
        ),
        (
            CulvertBarrel::vertical_ellipse(0.9652, 1.524, 0.419_475_92)?,
            34,
        ),
        (
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
            36,
        ),
        (
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
            37,
        ),
        (CulvertBarrel::arch(3.048, 1.6002)?, 46),
    ])
}

#[test]
fn arc_culvert_discharge_inverts_the_larger_control_headwater() -> TestResult {
    let mut lanes = Vec::new();
    let mut q = Vec::new();
    let mut tw = Vec::new();
    for (barrel, code) in barrels()? {
        let rise = barrel.rise_m();
        let inlet = CulvertInlet::hds5(code)?;
        // Mild and adverse within half the rise, with a short smooth barrel that leaves the
        // inlet in control.
        let mild = Culvert::both(barrel, [0.03, 0.0], inlet, 0.5);
        let adverse = Culvert::both(barrel, [0.0, 0.25 * rise], inlet, 0.5);
        let mut short = Culvert::both(barrel, [0.0, 0.0], inlet, 0.0);
        short.length_m = 2.0;
        short.manning_n = 0.009;
        let cases = [
            (mild, 0.3, 0.0),
            (mild, 2.0, 0.8 * rise),
            (mild, 4.0, 1.3 * rise),
            (adverse, 1.0, 0.0),
            (adverse, 3.0, 1.1 * rise),
            (short, 0.8, 0.0),
            (short, 3.0, 0.0),
        ];
        for (culvert, flow, tail) in cases {
            lanes.push(culvert);
            q.push(flow);
            tw.push(tail);
        }
    }
    let hw = control_headwater(&lanes, &q, &tw)?;
    let total: Vec<f64> = hw.iter().map(|h| h[0]).collect();
    let found = invert(&lanes, &total, &tw)?;
    let mut seen = [false; 2];
    for (index, ((discharge, regime, bounds), expected)) in found.iter().zip(&q).enumerate() {
        let [_, inlet, outlet] = hw[index];
        let what = format!(
            "lane {index} {:?}: Q={discharge:e}, expected {expected:e}, {regime:?}, bounds {bounds:?}, hw {:?}",
            lanes[index].barrel.shape(),
            hw[index]
        );
        assert!(close(*discharge, *expected, ROUND_TRIP_TOLERANCE), "{what}");
        if !close(inlet, outlet, SEPARATION) {
            let governing = if inlet > outlet {
                StructureRegime::CulvertInletControl
            } else {
                StructureRegime::CulvertOutletControl
            };
            assert_eq!(*regime, governing, "{what}");
            seen[usize::from(inlet < outlet)] = true;
        }
    }
    assert_eq!(seen, [true, true], "both controls govern some lane");
    Ok(())
}

#[test]
fn arc_culverts_hold_equal_heads_reverse_and_treat_dry_receivers_as_free_outfalls() -> TestResult {
    for (barrel, code) in barrels()? {
        let rise = barrel.rise_m();
        let inlet = CulvertInlet::hds5(code)?;
        let level = Culvert::both(barrel, [0.0, 0.0], inlet, 0.5);
        let culverts = [level; 8];
        let stage = 0.9 * rise;
        let supports = [
            (wet(stage), dry(0.0)),
            (dry(0.0), wet(stage)),
            (wet(stage), wet(stage)),
            (wet(stage), dry(-2.0)),
            (wet(stage), wet(0.0)),
            (wet(stage), wet(-0.4)),
            (wet(0.0), dry(0.0)),
            (dry(-0.5), dry(-0.5)),
        ];
        let outputs = propose(&elements(&culverts)?, &supports)?;
        let what = format!("{:?} code {code}: {outputs:?}", barrel.shape());
        let forward = outputs[0].discharge_m3_per_s;
        assert!(forward > 0.0, "{what}");
        assert_eq!(
            outputs[0].proposed_direction,
            Some(FlowDirection::AtoB),
            "{what}"
        );
        assert_eq!(
            outputs[1].discharge_m3_per_s.to_bits(),
            (-forward).to_bits(),
            "{what}"
        );
        assert_eq!(
            outputs[1].proposed_direction,
            Some(FlowDirection::BtoA),
            "{what}"
        );
        assert_eq!(
            (outputs[2].status, outputs[2].regime),
            (StructureStatus::PhysicalZero, StructureRegime::EqualHeads),
            "{what}"
        );
        for output in &outputs[3..6] {
            assert_eq!(
                output.discharge_m3_per_s.to_bits(),
                forward.to_bits(),
                "{what}"
            );
        }
        for output in &outputs[6..] {
            assert_eq!(
                output.discharge_m3_per_s.to_bits(),
                0.0f64.to_bits(),
                "{what}"
            );
            assert_eq!(
                (output.status, output.regime),
                (
                    StructureStatus::PhysicalZero,
                    StructureRegime::CulvertBelowInvert
                ),
                "{what}"
            );
        }
    }
    Ok(())
}

/// One evidence barrel per arc shape as constructor inputs, scaled by `2^power`.
fn scaled(index: usize, power: i32) -> TestResult<(CulvertBarrel, u8)> {
    let s = 2f64.powi(power);
    Ok(match index {
        0 => (
            CulvertBarrel::horizontal_ellipse(1.524 * s, 0.9652 * s, 0.419_475_92 * s)?,
            31,
        ),
        1 => (
            CulvertBarrel::vertical_ellipse(0.9652 * s, 1.524 * s, 0.419_475_92 * s)?,
            33,
        ),
        2 => (
            CulvertBarrel::pipe_arch(1.8542 * s, 1.397 * s, 0.4572 * s, 1.938_528 * s)?,
            41,
        ),
        _ => (CulvertBarrel::arch(3.048 * s, 1.6002 * s)?, 47),
    })
}

/// A mild one-way culvert of `barrel` with lengths times `2^power` and Manning n times
/// `2^(power/6)`, so friction slopes and Froude numbers stay as at unit scale.
fn similar(barrel: CulvertBarrel, code: u8, power: i32) -> TestResult<Culvert> {
    let s = 2f64.powi(power);
    let mut culvert = Culvert::both(barrel, [0.06 * s, 0.0], CulvertInlet::hds5(code)?, 0.5);
    culvert.length_m = 30.0 * s;
    culvert.manning_n = 0.013 * 2f64.powi(power / 6);
    culvert.end_b = None;
    Ok(culvert)
}

#[test]
fn scaled_arc_culverts_carry_the_unit_discharge_scaled() -> TestResult {
    // Heads scale by 2^60 and discharges by 2^150. Each scaled discharge meets the unit one
    // scaled within the round-trip tolerance.
    for index in 0..4 {
        let (unit, code) = scaled(index, 0)?;
        let rise = unit.rise_m();
        let heads = [0.5 * rise, 1.2 * rise, 2.5 * rise];
        let tails = [0.0, 0.3 * rise, 1.4 * rise];
        let lanes = vec![similar(unit, code, 0)?; heads.len()];
        let reference = invert(&lanes, &heads, &tails)?;
        for power in [-60, 60] {
            let (barrel, _) = scaled(index, power)?;
            let s = 2f64.powi(power);
            let lanes = vec![similar(barrel, code, power)?; heads.len()];
            let hw: Vec<f64> = heads.iter().map(|h| h * s).collect();
            let tw: Vec<f64> = tails.iter().map(|t| t * s).collect();
            let found = invert(&lanes, &hw, &tw)?;
            let flow = 2f64.powi(power * 5 / 2);
            for (lane, expected) in found.iter().zip(&reference) {
                assert_eq!(
                    lane.1, expected.1,
                    "{index} 2^{power}: {lane:?} {expected:?}"
                );
                assert!(
                    close(lane.0, expected.0 * flow, ROUND_TRIP_TOLERANCE),
                    "{index} 2^{power}: {lane:?} {expected:?}"
                );
            }
        }
        // At 2^300 the outlet march squares a discharge past F64, and the proposal says so.
        let (large, _) = scaled(index, 300)?;
        let s = 2f64.powi(300);
        let lanes = [similar(large, code, 300)?];
        match invert(&lanes, &[1.2 * rise * s], &[0.0]) {
            Err(error) => assert!(
                matches!(
                    error.downcast_ref::<StructureError>(),
                    Some(StructureError::Numerical { .. })
                ),
                "{index}: {error}"
            ),
            Ok(found) => return Err(format!("{index} 2^300: {found:?}").into()),
        }
        // At 2^-900 no represented discharge reaches the unsubmerged inlet band, and the
        // batch names the element.
        let (tiny, _) = scaled(index, -900)?;
        let elements = elements(&[similar(tiny, code, -900)?])?;
        match StructureBatch::<CpuRuntime>::with_culverts(client(), elements) {
            Err(StructureError::InvalidInput {
                kind: StructureInputErrorKind::Definition,
                context,
                element: Some(0),
            }) => assert!(
                context.contains("lies below the normal F64 range"),
                "{context}"
            ),
            other => return Err(format!("{index}: expected a rejection, found {other:?}").into()),
        }
    }
    Ok(())
}
