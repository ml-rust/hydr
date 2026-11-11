//! First-order rectangular channel flow through a caller-selected numerical runtime.

mod accounting;
mod admission;
mod config;
mod context;
mod definition;
mod engine;
mod error;
mod flux;
mod forcing;
mod friction;
mod interval;
mod output;
mod receipt;
mod reconstruction;
mod rounding;
mod snapshot;
mod sources;
mod stepping;
mod transport;

pub use config::ChannelConfig;
pub use definition::{
    ChannelBoundary, ChannelForcingInterval, ChannelInitialState, ChannelLateralBinding,
    ChannelLateralDirection, ChannelLateralInput, ChannelReachDefinition, RectangularChannelCell,
};
pub use engine::{ChannelReach, ChannelStep};
pub use error::ChannelError;
pub use output::{
    ChannelCellResult, ChannelLateralLedger, ChannelLateralReceipt, ChannelLateralStateData,
    ChannelStepDiagnostics, ChannelWaterLedger,
};
pub use snapshot::{ChannelIntervalSnapshotData, ChannelSnapshot, ChannelSnapshotData};
