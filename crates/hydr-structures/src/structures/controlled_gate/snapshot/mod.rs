//! Complete atomic trusted and caller-owned controller/actuator restoration.

mod check;
mod interval;
mod package;
mod restore;
mod state;

pub use package::{ControlledGateSnapshot, ControlledGateSnapshotData};
