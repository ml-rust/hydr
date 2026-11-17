//! Identity checks for native packages nested in composite checkpoints.

use crate::channel::{CHANNEL_CONTRACT_VERSION, CHANNEL_SNAPSHOT_SCHEMA, ChannelSnapshotData};
use crate::runoff::{
    SUBCATCHMENT_CONTRACT_VERSION, SUBCATCHMENT_SNAPSHOT_SCHEMA, SubcatchmentSnapshotData,
};
use crate::storage::{
    LEVEL_POOL_CONTRACT_VERSION, LEVEL_POOL_SNAPSHOT_SCHEMA, LevelPoolSnapshotData,
};
use hydr_core::contract::package_identity;

/// Check a nested level-pool package. The error message starts with `path`.
pub(crate) fn level_pool(path: &str, data: &LevelPoolSnapshotData) -> Result<(), String> {
    package_identity(
        "level pool",
        &data.schema,
        LEVEL_POOL_SNAPSHOT_SCHEMA,
        &data.contract,
        LEVEL_POOL_CONTRACT_VERSION,
    )
    .map_err(|message| format!("{path}: {message}"))
}

/// Check a nested channel package. The error message starts with `path`.
pub(crate) fn channel(path: &str, data: &ChannelSnapshotData) -> Result<(), String> {
    package_identity(
        "channel",
        &data.schema,
        CHANNEL_SNAPSHOT_SCHEMA,
        &data.contract,
        CHANNEL_CONTRACT_VERSION,
    )
    .map_err(|message| format!("{path}: {message}"))
}

/// Check a nested subcatchment package. The error message starts with `path`.
pub(crate) fn subcatchment(path: &str, data: &SubcatchmentSnapshotData) -> Result<(), String> {
    package_identity(
        "subcatchment",
        &data.schema,
        SUBCATCHMENT_SNAPSHOT_SCHEMA,
        &data.contract,
        SUBCATCHMENT_CONTRACT_VERSION,
    )
    .map_err(|message| format!("{path}: {message}"))
}
