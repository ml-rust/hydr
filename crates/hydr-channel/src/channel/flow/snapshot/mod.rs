//! Complete bounded continuation packages for rectangular channel reaches.

mod check;
#[cfg(test)]
mod fixture;
mod package;
mod restore;
mod trusted;

pub use package::{
    CHANNEL_CONTRACT_VERSION, CHANNEL_SNAPSHOT_SCHEMA, ChannelIntervalSnapshotData,
    ChannelSnapshot, ChannelSnapshotData,
};
