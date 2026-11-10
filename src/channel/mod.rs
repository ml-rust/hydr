//! Declared channel geometry and rectangular reach flow for hydraulic applications.

mod flow;
mod section;

pub use flow::{
    ChannelBoundary, ChannelBoundaryInterval, ChannelCellResult, ChannelConfig, ChannelError,
    ChannelInitialState, ChannelIntervalSnapshotData, ChannelReach, ChannelReachDefinition,
    ChannelSnapshot, ChannelSnapshotData, ChannelStep, ChannelStepDiagnostics, ChannelWaterLedger,
    RectangularChannelCell,
};
pub use section::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionError, CrossSectionPoint,
    CrossSectionProperties, CrossSectionQuery,
};
