//! Channel flow through a caller-selected numerical runtime.

mod accounting;
mod admission;
mod boundary;
mod boundary_config;
mod boundary_trace;
mod config;
mod conserved_point;
mod conserved_update;
mod context;
mod coupling;
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
mod surveyed_recovery;
mod surveyed_residuals;
#[cfg(test)]
mod surveyed_state_tests;
mod surveyed_transport;
mod surveyed_update;
mod transport;
mod triangle_reconstruction;
mod triangular;

pub use boundary_config::ChannelBoundaryConfig;
pub use boundary_trace::{ChannelBoundaryTrace, ChannelReservoirTrace};
pub use config::{ChannelConfig, ChannelTriangularReconstruction};
pub use definition::{
    ChannelBoundary, ChannelForcingInterval, ChannelGeometry, ChannelInitialState,
    ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput, ChannelReachDefinition,
    RectangularChannelCell, TriangularChannelCell,
};
pub use engine::{ChannelReach, ChannelStep};
pub use error::ChannelError;
pub use geometry::{ChannelSurveyedConfig, SurveyedChannelCell};
pub use output::{
    ChannelCellResult, ChannelFrictionGeometryDiagnostics, ChannelLateralLedger,
    ChannelLateralReceipt, ChannelLateralStateData, ChannelRiemannDiagnostics,
    ChannelStepDiagnostics, ChannelWaterLedger,
};
pub use snapshot::{ChannelIntervalSnapshotData, ChannelSnapshot, ChannelSnapshotData};

pub use forcing::{
    ChannelTensorBoundary, ChannelTensorForcingInterval, ChannelTensorLateralInput,
    ChannelTensorWaterAmount,
};

pub(crate) use coupling::{ChannelCouplingForcing, ChannelCouplingTrial, ChannelReservoirOperands};

pub(crate) mod surveyed;
#[cfg(test)]
pub(crate) mod surveyed_points;
pub use surveyed::{
    PrismaticRiemannConfig, PrismaticRiemannError, PrismaticRiemannRejection,
    PrismaticRiemannResult, PrismaticRiemannWork,
};
