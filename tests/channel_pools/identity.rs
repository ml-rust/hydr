//! Channel-pool package schema and contract identity, including nested packages.

use super::fixture::{Batch, Result};
use super::restoration::history;
use hydr::channel::{CHANNEL_CONTRACT_VERSION, CHANNEL_SNAPSHOT_SCHEMA};
use hydr::coupling::{
    CHANNEL_POOL_CONTRACT_VERSION, CHANNEL_POOL_SNAPSHOT_SCHEMA, ChannelPoolError,
    ChannelPoolSnapshotData,
};
use hydr::storage::LEVEL_POOL_CONTRACT_VERSION;

/// Import `changed` into `origin`, then check the input error names every `needle`.
/// Import returns a new batch, so `origin` must keep its original package.
fn assert_rejected(origin: &Batch, changed: &ChannelPoolSnapshotData, needles: &[&str]) -> Result {
    let before = origin.snapshot_data()?;
    let Err(error) = origin.import_snapshot(changed) else {
        return Err("foreign identity was accepted".into());
    };
    assert!(
        matches!(error, ChannelPoolError::Input(_)),
        "unexpected kind: {error}"
    );
    let message = error.to_string();
    for needle in needles {
        assert!(message.contains(needle), "missing {needle:?}: {message}");
    }
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

/// Package with retained history, so import runs the replay path.
fn package() -> Result<(Batch, ChannelPoolSnapshotData)> {
    let (origin, current) = history()?;
    let package = current.snapshot_data()?;
    if package.last_interval.is_none() {
        return Err("history package lacks its final interval".into());
    }
    Ok((origin, package))
}

#[test]
fn channel_pool_package_carries_identity() -> Result {
    assert_eq!(
        CHANNEL_POOL_SNAPSHOT_SCHEMA,
        "hydr-channel-pool-checkpoint-1"
    );
    assert_eq!(
        CHANNEL_POOL_CONTRACT_VERSION,
        "hydr-channel-pool-numr-cpu-1"
    );
    let (_, package) = package()?;
    assert_eq!(package.schema, CHANNEL_POOL_SNAPSHOT_SCHEMA);
    assert_eq!(package.contract, CHANNEL_POOL_CONTRACT_VERSION);
    Ok(())
}

#[test]
fn channel_pool_import_rejects_foreign_schema() -> Result {
    let (origin, package) = package()?;
    for found in ["hydr-channel-pool-checkpoint-0", ""] {
        let mut changed = package.clone();
        changed.schema = found.into();
        assert_rejected(
            &origin,
            &changed,
            &[
                "channel pool checkpoint schema",
                &format!("found \"{found}\""),
                CHANNEL_POOL_SNAPSHOT_SCHEMA,
                "capture a new package",
            ],
        )?;
    }
    Ok(())
}

#[test]
fn channel_pool_import_rejects_foreign_contract() -> Result {
    let (origin, mut changed) = package()?;
    changed.contract = "hydr-channel-pool-numr-gpu-1".into();
    assert_rejected(
        &origin,
        &changed,
        &[
            "channel pool checkpoint contract",
            "found \"hydr-channel-pool-numr-gpu-1\"",
            CHANNEL_POOL_CONTRACT_VERSION,
        ],
    )
}

#[test]
fn channel_pool_import_rejects_nested_reach_schema() -> Result {
    let (origin, mut changed) = package()?;
    let id = format!("{:?}", changed.setup.reaches[0].id);
    changed.reaches[0].schema = "hydr-channel-checkpoint-0".into();
    assert_rejected(
        &origin,
        &changed,
        &[
            &format!("channel pool checkpoint reaches[0] ({id})"),
            "channel checkpoint schema",
            "found \"hydr-channel-checkpoint-0\"",
            CHANNEL_SNAPSHOT_SCHEMA,
        ],
    )
}

#[test]
fn channel_pool_import_rejects_nested_previous_reach_schema() -> Result {
    let (origin, mut changed) = package()?;
    let id = format!("{:?}", changed.setup.reaches[0].id);
    let last = changed
        .last_interval
        .as_mut()
        .ok_or("history package lacks its final interval")?;
    last.previous.reaches[0].schema = "hydr-channel-checkpoint-0".into();
    assert_rejected(
        &origin,
        &changed,
        &[
            &format!("channel pool checkpoint last_interval.previous.reaches[0] ({id})"),
            "channel checkpoint schema",
            "found \"hydr-channel-checkpoint-0\"",
            CHANNEL_SNAPSHOT_SCHEMA,
        ],
    )
}

#[test]
fn channel_pool_import_rejects_nested_previous_reach_contract() -> Result {
    let (origin, mut changed) = package()?;
    let last = changed
        .last_interval
        .as_mut()
        .ok_or("history package lacks its final interval")?;
    last.previous.reaches[0].contract = "hydr-channel-1d-numr-gpu-1".into();
    assert_rejected(
        &origin,
        &changed,
        &[
            "channel pool checkpoint last_interval.previous.reaches[0]",
            "channel checkpoint contract",
            "found \"hydr-channel-1d-numr-gpu-1\"",
            CHANNEL_CONTRACT_VERSION,
        ],
    )
}

#[test]
fn channel_pool_import_rejects_nested_previous_pools_contract() -> Result {
    let (origin, mut changed) = package()?;
    let last = changed
        .last_interval
        .as_mut()
        .ok_or("history package lacks its final interval")?;
    last.previous.pools.contract = "hydr-level-pool-numr-gpu-1".into();
    assert_rejected(
        &origin,
        &changed,
        &[
            "channel pool checkpoint last_interval.previous.pools",
            "level pool checkpoint contract",
            "found \"hydr-level-pool-numr-gpu-1\"",
            LEVEL_POOL_CONTRACT_VERSION,
        ],
    )
}

#[test]
fn channel_pool_import_rejects_nested_pools_contract() -> Result {
    let (origin, mut changed) = package()?;
    changed.pools.contract = "hydr-level-pool-numr-gpu-1".into();
    assert_rejected(
        &origin,
        &changed,
        &[
            "channel pool checkpoint pools",
            "level pool checkpoint contract",
            "found \"hydr-level-pool-numr-gpu-1\"",
            LEVEL_POOL_CONTRACT_VERSION,
        ],
    )
}
