//! Channel flow through a caller-selected numerical runtime.

mod accounting;
mod admission;
mod boundary;
mod boundary_config;
mod boundary_trace;
mod config;
mod context;
mod definition;
mod engine;
mod error;
mod flux;
mod forcing;
mod friction;
mod geometry;
mod interval;
mod output;
mod receipt;
mod reconstruction;
mod riemann_diagnostics;
mod rounding;
mod snapshot;
mod sources;
mod stepping;
mod transport;
mod triangle_reconstruction;
mod triangular;

pub use boundary_config::ChannelBoundaryConfig;
pub use boundary_trace::ChannelBoundaryTrace;
pub use config::{ChannelConfig, ChannelTriangularReconstruction};
pub use definition::{
    ChannelBoundary, ChannelForcingInterval, ChannelGeometry, ChannelInitialState,
    ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput, ChannelReachDefinition,
    RectangularChannelCell, TriangularChannelCell,
};
pub use engine::{ChannelReach, ChannelStep};
pub use error::ChannelError;
pub use output::{
    ChannelCellResult, ChannelLateralLedger, ChannelLateralReceipt, ChannelLateralStateData,
    ChannelRiemannDiagnostics, ChannelStepDiagnostics, ChannelWaterLedger,
};
pub use snapshot::{ChannelIntervalSnapshotData, ChannelSnapshot, ChannelSnapshotData};
