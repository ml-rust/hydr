//! Complete trusted and caller-owned sampled-control restoration.

mod compatibility;
mod package;
mod receipt_bits;
mod restore;
mod state_check;
mod transition_check;

pub use package::{OperatingControlSnapshot, OperatingControlSnapshotData};
