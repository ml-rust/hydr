//! Complete bounded continuation packages for rectangular channel reaches.

mod check;
mod package;
mod restore;

#[cfg(test)]
mod tests;

pub use package::{
    CHANNEL_CONTRACT_VERSION, CHANNEL_SNAPSHOT_SCHEMA, ChannelIntervalSnapshotData,
    ChannelSnapshot, ChannelSnapshotData,
};
