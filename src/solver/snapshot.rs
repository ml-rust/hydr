//! Typed checkpoint packages and checked accepted-state import.

mod boundary;
mod cfl;
mod check;
mod package;
mod transmissive;

pub use package::{SNAPSHOT_SCHEMA, Snapshot, SnapshotData};
