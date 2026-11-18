//! Identity checks for native packages nested in composite checkpoints.

use crate::channel::{CHANNEL_CONTRACT_VERSION, CHANNEL_SNAPSHOT_SCHEMA, ChannelSnapshotData};
use crate::runoff::{
    SUBCATCHMENT_CONTRACT_VERSION, SUBCATCHMENT_SNAPSHOT_SCHEMA, SubcatchmentSnapshotData,
};
use crate::solver::{CONTRACT_VERSION, SNAPSHOT_SCHEMA, SnapshotData};
use crate::storage::{
    LEVEL_POOL_CONTRACT_VERSION, LEVEL_POOL_SNAPSHOT_SCHEMA, LevelPoolSnapshotData,
};
use crate::structures::{
    CONTROLLED_GATE_CONTRACT_VERSION, CONTROLLED_GATE_SNAPSHOT_SCHEMA,
    CONTROLLED_PUMP_CONTRACT_VERSION, CONTROLLED_PUMP_SNAPSHOT_SCHEMA, ControlledGateSnapshotData,
    ControlledPumpSnapshotData, GATE_ACTUATOR_CONTRACT_VERSION, GATE_ACTUATOR_SNAPSHOT_SCHEMA,
    OPERATING_CONTROL_CONTRACT_VERSION, OPERATING_CONTROL_SNAPSHOT_SCHEMA,
    OperatingControlSnapshotData,
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

/// Check a nested controlled-gate package and its control and actuator packages.
///
/// The error message starts with `path`.
pub(crate) fn controlled_gate(path: &str, data: &ControlledGateSnapshotData) -> Result<(), String> {
    package_identity(
        "controlled gate",
        &data.schema,
        CONTROLLED_GATE_SNAPSHOT_SCHEMA,
        &data.contract,
        CONTROLLED_GATE_CONTRACT_VERSION,
    )
    .map_err(|message| format!("{path}: {message}"))?;
    operating_control(&format!("{path}.controls"), &data.controls)?;
    package_identity(
        "gate actuator",
        &data.actuators.schema,
        GATE_ACTUATOR_SNAPSHOT_SCHEMA,
        &data.actuators.contract,
        GATE_ACTUATOR_CONTRACT_VERSION,
    )
    .map_err(|message| format!("{path}.actuators: {message}"))
}

/// Check a nested controlled-pump package and its control package.
///
/// The error message starts with `path`.
pub(crate) fn controlled_pump(path: &str, data: &ControlledPumpSnapshotData) -> Result<(), String> {
    package_identity(
        "controlled pump",
        &data.schema,
        CONTROLLED_PUMP_SNAPSHOT_SCHEMA,
        &data.contract,
        CONTROLLED_PUMP_CONTRACT_VERSION,
    )
    .map_err(|message| format!("{path}: {message}"))?;
    operating_control(&format!("{path}.controls"), &data.controls)
}

/// Check a nested operating-control package. The error message starts with `path`.
fn operating_control(path: &str, data: &OperatingControlSnapshotData) -> Result<(), String> {
    package_identity(
        "operating control",
        &data.schema,
        OPERATING_CONTROL_SNAPSHOT_SCHEMA,
        &data.contract,
        OPERATING_CONTROL_CONTRACT_VERSION,
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

/// Check a nested surface solver package. The error message starts with `path`.
pub(crate) fn surface(path: &str, data: &SnapshotData) -> Result<(), String> {
    package_identity(
        "surface solver",
        &data.schema,
        SNAPSHOT_SCHEMA,
        &data.contract,
        CONTRACT_VERSION,
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
