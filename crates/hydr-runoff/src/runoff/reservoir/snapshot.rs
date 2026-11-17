//! Complete reservoir continuation packages and atomic checked restoration.

mod package;
mod receipts;
mod restore;

pub use package::{
    RESERVOIR_CONTRACT_VERSION, RESERVOIR_SNAPSHOT_SCHEMA, ReservoirSnapshot, ReservoirSnapshotData,
};
