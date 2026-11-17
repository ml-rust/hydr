//! Complete atomic trusted and caller-owned controller/actuator restoration.

mod check;
mod interval;
mod package;
mod restore;
mod state;

pub use package::{
    CONTROLLED_GATE_CONTRACT_VERSION, CONTROLLED_GATE_SNAPSHOT_SCHEMA, ControlledGateSnapshot,
    ControlledGateSnapshotData,
};
