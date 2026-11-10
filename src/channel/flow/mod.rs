//! First-order rectangular channel flow through a caller-selected numerical runtime.

mod accounting;
mod admission;
mod config;
mod context;
mod definition;
mod engine;
mod error;
mod flux;
mod friction;
mod output;
mod receipt;
mod rounding;
mod snapshot;
mod stepping;

pub use config::ChannelConfig;
pub use definition::{
    ChannelBoundary, ChannelBoundaryInterval, ChannelInitialState, ChannelReachDefinition,
    RectangularChannelCell,
};
pub use engine::{ChannelReach, ChannelStep};
pub use error::ChannelError;
pub use output::{ChannelCellResult, ChannelStepDiagnostics, ChannelWaterLedger};
pub use snapshot::{ChannelIntervalSnapshotData, ChannelSnapshot, ChannelSnapshotData};
