//! Pooled checkpoints on a surveyed reach: bitwise restart and forged pool-end packages.
//!
//! Frozen criterion, "Coordinated restart"): the case of `pools_restart.rs`
//! with the reach a surveyed rectangle, whose right end is a pool end into the pond.
//!
//! - Snapshot and restore, import into a fresh batch, and resume each match the checkpoint
//!   state bit for bit.
//! - Each continues `AFTER` intervals and matches the uninterrupted run bit for bit.
//! - A forged pool-end ledger, pool-end receipt, pool package, or structure ledger rejects in
//!   import and in resume.

use crate::pools_fixture::{Batch, Result, state, surveyed_reach};
use crate::pools_restart::{AFTER, case, run, to_checkpoint};
use hydr::coupling::{SurfaceChannelError, SurfaceChannelSnapshotData};

/// The batch of the restart case with its reach cell surveyed.
fn surveyed_batch() -> Result<Batch> {
    let case = case()?;
    let (mut setup, initial) = case.parts()?;
    setup.reaches[0].definition = surveyed_reach(case.reach_bed);
    Ok(Batch::new(crate::runtime::client(), setup, initial)?)
}

/// Input messages of import into a fresh batch and of resume, in that order.
fn rejections(data: &SurfaceChannelSnapshotData) -> Result<[String; 2]> {
    let message = |result: std::result::Result<Batch, SurfaceChannelError>| match result {
        Ok(_) => Err("the coordinator accepted a forged package".into()),
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
    };
    let imported: Result<String> = message(surveyed_batch()?.import_snapshot(data));
    let resumed: Result<String> = message(Batch::resume(
        crate::runtime::client(),
        data,
        data.surface.setup.forcing.clone(),
    ));
    Ok([imported?, resumed?])
}

/// Fails unless import and resume both reject `data` with every fragment.
fn rejects(data: &SurfaceChannelSnapshotData, fragments: &[&str]) -> Result {
    for message in rejections(data)? {
        for fragment in fragments {
            assert!(message.contains(fragment), "{message:?} lacks {fragment:?}");
        }
    }
    Ok(())
}

#[test]
fn surveyed_every_restart_path_matches_the_uninterrupted_run() -> Result {
    let checkpoint = to_checkpoint(surveyed_batch()?)?;
    let ends = checkpoint.pool_end_ledgers()?;
    assert!(
        ends.iter()
            .any(|e| e.reach_to_pool_m3 > 0.0 || e.pool_to_reach_m3 > 0.0)
    );
    let expected = state(&checkpoint)?;
    let data = checkpoint.snapshot_data()?;
    let restored = surveyed_batch()?.restore(&checkpoint.snapshot())?;
    let imported = surveyed_batch()?.import_snapshot(&data)?;
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

#[test]
fn surveyed_zero_history_package_restores_the_fresh_batch() -> Result {
    let fresh = surveyed_batch()?;
    let data = fresh.snapshot_data()?;
    let expected = state(&fresh)?;
    assert_eq!(state(&surveyed_batch()?.import_snapshot(&data)?)?, expected);
    let resumed = Batch::resume(
        crate::runtime::client(),
        &data,
        data.surface.setup.forcing.clone(),
    )?;
    assert_eq!(state(&resumed)?, expected);
    Ok(())
}

#[test]
fn surveyed_forged_pool_end_pool_and_structure_packages_reject() -> Result {
    let data = to_checkpoint(surveyed_batch()?)?.snapshot_data()?;
    let mut forged = data.clone();
    let pools = forged.pools.as_mut().ok_or("the pool package")?;
    pools.states[0].volume_m3 += 1e-6;
    rejections(&forged)?;
    let mut forged = data.clone();
    forged.counters.pool_ends[0].reach_to_pool_m3 += 1e-6;
    rejects(&forged, &["counters"])?;
    let mut forged = data.clone();
    forged.counters.structures[0].a_to_b_m3 += 1e-6;
    rejects(&forged, &["counters"])?;
    let mut forged = data.clone();
    let last = forged.last_interval.as_mut().ok_or("the last interval")?;
    last.diagnostics.pool_ends[0].signed_volume_m3 += 1e-6;
    rejects(&forged, &["last_interval.diagnostics.pool_ends"])?;
    let mut forged = data;
    let last = forged.last_interval.as_mut().ok_or("the last interval")?;
    last.diagnostics.structures[0].law_volume_m3 += 1e-6;
    rejects(
        &forged,
        &[
            "last_interval.diagnostics.structures[0]",
            "differs from the recalculated last interval",
        ],
    )
}
