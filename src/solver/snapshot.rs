//! Typed checkpoint packages and checked accepted-state import.

mod boundary;
mod check;
mod package;

pub use package::{SNAPSHOT_SCHEMA, Snapshot, SnapshotData};
