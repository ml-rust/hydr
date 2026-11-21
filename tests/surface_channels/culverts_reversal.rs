//! A tide on a reach cell floods and drains a lagoon through two culverts.
//!
//! Frozen criteria, fixed before the first run:
//! - Every interval passes the mass, pairing, and composite checks. The reach step limit can
//!   shorten an interval. No interval crosses a tide change.
//! - The two-way culvert carries water both ways over the run: into the reach and into the
//!   lagoon. A flowing interval reports outlet control.
//! - The one-end culvert carries the two-way culvert's sign whenever that culvert flows from
//!   the lagoon. Whenever it flows into the lagoon, the one-end culvert reports
//!   `CulvertUndeclaredEnd` and moves `0.0` bit for bit.

use crate::culverts_fixture::{reversal_case, tide, tide_rate};
use crate::pools_evidence::check;
use crate::pools_fixture::{Batch, Result, transfer};
use hydr::structures::StructureRegime;

/// Requested duration of every interval, in s.
const DT: f64 = 0.5;

/// End of the run, in s: a flood, an ebb, and a second flood.
const END_S: f64 = 16.0;

/// `batch` after one tide interval of at most `DT`, ending at or before the next tide change.
pub fn tide_step(batch: &Batch) -> Result<Batch> {
    let start = batch.time();
    let end = TIDE_CHANGES
        .into_iter()
        .find(|&change| change > start)
        .map_or(start + DT, |change| change.min(start + DT));
    let forcing = tide(batch.setup(), (start, end), tide_rate(start));
    let next = batch.trial_to(end, &forcing)?.into_next();
    check(&next, &[])?;
    Ok(next)
}

/// Times the tide rate changes, in s, and the end of the run.
const TIDE_CHANGES: [f64; 3] = [4.0, 12.0, END_S];

#[test]
fn a_tide_reverses_a_two_way_culvert_and_closes_a_one_end_culvert() -> Result {
    let mut batch = reversal_case()?.build()?;
    let (mut to_lagoon, mut from_lagoon) = (false, false);
    while batch.time() < END_S {
        batch = tide_step(&batch)?;
        let two_way = transfer(&batch, "tide")?;
        let flap = transfer(&batch, "flap")?;
        if two_way.signed_volume_m3 != 0.0 {
            assert_eq!(
                two_way.regime,
                StructureRegime::CulvertOutletControl,
                "{two_way:?}"
            );
        }
        if two_way.signed_volume_m3 > 0.0 {
            from_lagoon = true;
            assert!(flap.signed_volume_m3 > 0.0, "{flap:?}");
        }
        if two_way.signed_volume_m3 < 0.0 {
            to_lagoon = true;
            assert_eq!(
                flap.regime,
                StructureRegime::CulvertUndeclaredEnd,
                "{flap:?}"
            );
            assert_eq!(
                flap.signed_volume_m3.to_bits(),
                0.0f64.to_bits(),
                "{flap:?}"
            );
        }
    }
    assert!(
        to_lagoon && from_lagoon,
        "the tide never reversed the culvert"
    );
    Ok(())
}
