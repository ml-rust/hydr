//! Complete bounded continuation packages for rectangular channel reaches.

mod check;
mod package;
mod restore;

#[cfg(test)]
mod tests;

pub use package::{ChannelIntervalSnapshotData, ChannelSnapshot, ChannelSnapshotData};
