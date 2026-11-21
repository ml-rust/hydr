//! An upper pond drains through a culvert and over the road above it into a lower pond.
//!
//! Frozen criteria, fixed before the first run:
//! - Every interval passes the mass, pairing, and composite checks.
//! - While the upper pond stands above the road crest, the road weir flows freely beside the
//!   drowned culvert. Once it falls to the crest, the weir stops and the culvert flows on.
//! - Snapshot and restore, import into a fresh batch, and resume each match the checkpoint
//!   state bit for bit. Each continues `AFTER` intervals and matches the uninterrupted run bit
//!   for bit, package, every ledger, and every child state included.

use crate::culverts_fixture::{ROAD_CREST_M, roadway_case};
use crate::pools_evidence::{check, pool_stage};
use crate::pools_fixture::{Batch, PoolForcing, Result, state, step, transfer};
use hydr::structures::StructureRegime;

/// Requested duration of every interval, in s.
const DT: f64 = 0.5;

/// Intervals before the checkpoint.
const BEFORE: usize = 4;

/// Intervals after the checkpoint.
const AFTER: usize = 4;

/// `batch` after `count` checked intervals of `DT`.
fn run(mut batch: Batch, count: usize) -> Result<Batch> {
    for _ in 0..count {
        batch = step(&batch, DT, &PoolForcing::default())?;
        check(&batch, &[])?;
    }
    Ok(batch)
}

#[test]
fn a_road_overtops_beside_a_drowned_culvert_until_the_crest() -> Result {
    let mut batch = roadway_case()?.build()?;
    let (mut overtopped, mut below) = (false, false);
    for _ in 0..40 {
        batch = run(batch, 1)?;
        let culvert = transfer(&batch, "culvert")?;
        let road = transfer(&batch, "road")?;
        assert_eq!(
            culvert.regime,
            StructureRegime::CulvertOutletControl,
            "{culvert:?}"
        );
        assert!(culvert.signed_volume_m3 > 0.0, "{culvert:?}");
        if road.midpoint_stage_a_m > ROAD_CREST_M {
            overtopped = true;
            assert_eq!(road.regime, StructureRegime::FreeWeirOverflow, "{road:?}");
            assert!(road.signed_volume_m3 > 0.0, "{road:?}");
        } else {
            below = true;
            assert_eq!(
                road.signed_volume_m3.to_bits(),
                0.0f64.to_bits(),
                "{road:?}"
            );
        }
    }
    assert!(
        overtopped && below,
        "the road never overtopped or never stopped"
    );
    assert!(pool_stage(&batch, "upper")? <= ROAD_CREST_M);
    Ok(())
}

#[test]
fn every_restart_path_matches_the_uninterrupted_culvert_run() -> Result {
    let checkpoint = run(roadway_case()?.build()?, BEFORE)?;
    assert!(transfer(&checkpoint, "road")?.signed_volume_m3 > 0.0);
    let expected = state(&checkpoint)?;
    let data = checkpoint.snapshot_data()?;
    let restored = roadway_case()?.build()?.restore(&checkpoint.snapshot())?;
    let imported = roadway_case()?.build()?.import_snapshot(&data)?;
    let resumed = Batch::resume(
        crate::runtime::client(),
        &data,
        data.surface.setup.forcing.clone(),
    )?;
    let uninterrupted = state(&run(checkpoint, AFTER)?)?;
    for (path, batch) in [
        ("restore", restored),
        ("import", imported),
        ("resume", resumed),
    ] {
        assert_eq!(state(&batch)?, expected, "{path} at the checkpoint");
        assert_eq!(
            state(&run(batch, AFTER)?)?,
            uninterrupted,
            "{path} after the checkpoint"
        );
    }
    Ok(())
}
