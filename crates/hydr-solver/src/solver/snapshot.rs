//! Typed checkpoint packages and checked accepted-state import.

mod boundary;
mod cfl;
mod check;
mod forcing_window;
mod last_step;
mod package;
mod soil_receipt;
mod sources;
mod transmissive;

pub(in crate::solver) use check::{Exchange, reject_exchange};
pub use package::{SNAPSHOT_SCHEMA, Snapshot, SnapshotData};
