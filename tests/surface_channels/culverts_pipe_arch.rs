//! A head pond drains through a pipe-arch culvert into a dry basin.
//!
//! Frozen criteria, fixed before the first run:
//! - Every interval passes the mass, pairing, and composite checks.
//! - The culvert carries water into the basin every interval under inlet or outlet control.
//! - The two ponds together hold their starting volume within `VOLUME_RELATIVE`.
//! - Snapshot and restore, import into a fresh batch, and resume each match the checkpoint
//!   state bit for bit. Each continues `AFTER` intervals and matches the uninterrupted run bit
//!   for bit, package, every ledger, and every child state included.

use crate::culverts_fixture::pipe_arch_case;
use crate::pools_evidence::{check, pool_volume};
use crate::pools_fixture::{Batch, PoolForcing, Result, state, step, transfer};
use hydr::structures::StructureRegime;

/// Requested duration of every interval, in s.
const DT: f64 = 0.5;

/// Intervals of the drain run.
const INTERVALS: usize = 8;

/// Intervals before the checkpoint.
const BEFORE: usize = 3;

/// Intervals after the checkpoint.
const AFTER: usize = 3;

/// Starting volume of the head pond, the basin starting dry, in m³.
const START_M3: f64 = 200.0;

/// Largest relative change of the two ponds' total volume.
const VOLUME_RELATIVE: f64 = 1e-12;

/// `batch` after `count` checked intervals of `DT`.
fn run(mut batch: Batch, count: usize) -> Result<Batch> {
    for _ in 0..count {
        batch = step(&batch, DT, &PoolForcing::default())?;
        check(&batch, &[])?;
    }
    Ok(batch)
}

#[test]
fn a_pipe_arch_culvert_drains_a_head_pond_and_conserves_volume() -> Result {
    let mut batch = pipe_arch_case()?.build()?;
    let mut basin = 0.0;
    for _ in 0..INTERVALS {
        batch = run(batch, 1)?;
        let culvert = transfer(&batch, "culvert")?;
        assert!(
            matches!(
                culvert.regime,
                StructureRegime::CulvertInletControl | StructureRegime::CulvertOutletControl
            ),
            "{culvert:?}"
        );
        assert!(culvert.signed_volume_m3 > 0.0, "{culvert:?}");
        let filled = pool_volume(&batch, "basin")?;
        assert!(filled > basin, "basin {filled} after {basin}");
        basin = filled;
        let total = pool_volume(&batch, "head")? + filled;
        assert!(
            (total - START_M3).abs() <= VOLUME_RELATIVE * START_M3,
            "ponds hold {total} m³"
        );
    }
    Ok(())
}

#[test]
fn every_restart_path_matches_the_uninterrupted_pipe_arch_run() -> Result {
    let checkpoint = run(pipe_arch_case()?.build()?, BEFORE)?;
    let expected = state(&checkpoint)?;
    let data = checkpoint.snapshot_data()?;
    let restored = pipe_arch_case()?.build()?.restore(&checkpoint.snapshot())?;
    let imported = pipe_arch_case()?.build()?.import_snapshot(&data)?;
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
