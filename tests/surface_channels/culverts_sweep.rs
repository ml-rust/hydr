//! A filling basin drowns a culvert outlet, so control passes from the inlet to the outlet.
//!
//! Frozen criteria, fixed before the first run:
//! - Every interval passes the mass, pairing, and composite checks.
//! - The culvert starts under inlet control into the dry basin and ends under outlet control.
//!   Once outlet control or a tie appears, inlet control never returns.
//! - The discharge falls as the basin fills. Between neighbouring intervals the mean discharge
//!   changes by at most `JUMP` of the larger value, across the change of control included.

use crate::culverts_fixture::{LONG_RISE_M, drowning_case, sweep_case};
use crate::pools_evidence::{check, pool_stage};
use crate::pools_fixture::{Batch, Case, PoolForcing, Result, forcing, transfer};
use hydr::structures::StructureRegime;

/// `case` advanced to `end` in intervals of at most `DT`, each checked, with the mean culvert
/// discharge and regime of every interval.
fn sweep(case: &Case, end_s: f64) -> Result<(Batch, Vec<f64>, Vec<StructureRegime>)> {
    let mut batch = case.build()?;
    let mut regimes = Vec::new();
    let mut discharges: Vec<f64> = Vec::new();
    while batch.time() < end_s {
        let end = (batch.time() + DT).min(end_s);
        let held = forcing(&batch, end, &PoolForcing::default());
        batch = batch.trial_to(end, &held)?.into_next();
        check(&batch, &[])?;
        let step = batch.last_step()?.ok_or("an accepted interval")?;
        let culvert = transfer(&batch, "culvert")?;
        assert!(culvert.signed_volume_m3 > 0.0, "{culvert:?}");
        discharges.push(culvert.signed_volume_m3 / (step.end_s - step.start_s));
        regimes.push(culvert.regime);
    }
    Ok((batch, discharges, regimes))
}

/// Fails unless no neighbouring pair of `discharges` differs by more than `JUMP` of the larger.
fn continuous(discharges: &[f64]) {
    for pair in discharges.windows(2) {
        let larger = pair[0].max(pair[1]);
        assert!(
            (pair[0] - pair[1]).abs() <= JUMP * larger,
            "discharges={discharges:?}"
        );
    }
}

/// Requested duration of every interval, in s.
const DT: f64 = 0.5;

/// End of the run, in s.
const END_S: f64 = 10.0;

/// Largest relative change of the mean discharge between neighbouring intervals.
const JUMP: f64 = 0.25;

#[test]
fn a_filling_basin_moves_a_culvert_from_inlet_to_outlet_control() -> Result {
    let (_, discharges, regimes) = sweep(&sweep_case()?, END_S)?;
    println!("culvert sweep: regimes={regimes:?}, discharges={discharges:?}");
    for pair in discharges.windows(2) {
        assert!(pair[1] < pair[0], "discharges={discharges:?}");
    }
    continuous(&discharges);
    assert_eq!(regimes.first(), Some(&StructureRegime::CulvertInletControl));
    assert_eq!(regimes.last(), Some(&StructureRegime::CulvertOutletControl));
    let switch = regimes
        .iter()
        .position(|r| *r != StructureRegime::CulvertInletControl)
        .ok_or("no change of control")?;
    for regime in &regimes[switch..] {
        assert!(
            matches!(
                regime,
                StructureRegime::CulvertOutletControl | StructureRegime::CulvertControlTie
            ),
            "{regimes:?}"
        );
    }
    Ok(())
}

/// End of the drowning run, in s.
const DROWNING_END_S: f64 = 6.0;

/// Frozen criteria, fixed before the first run:
/// - Every interval passes the mass, pairing, and composite checks and reports outlet control.
/// - The basin starts below the barrel crown, so the outlet runs partly full, and ends above
///   it, so the barrel runs full.
/// - Between neighbouring intervals the mean discharge changes by at most `JUMP` of the larger
///   value, across the crown included.
#[test]
fn a_filling_basin_drowns_a_partly_full_culvert_outlet() -> Result {
    let (batch, discharges, regimes) = sweep(&drowning_case()?, DROWNING_END_S)?;
    println!("culvert drowning: regimes={regimes:?}, discharges={discharges:?}");
    for regime in &regimes {
        assert_eq!(
            *regime,
            StructureRegime::CulvertOutletControl,
            "{regimes:?}"
        );
    }
    assert!(pool_stage(&batch, "basin")? > LONG_RISE_M);
    continuous(&discharges);
    Ok(())
}
