//! Simultaneous 3-way conservative coupling of 2D surface flow, 1D river channels (with pools,
//! structures, and runoff), and pinned EPA SWMM v5.2.4 piped urban drainage.
//!
//! [`SurfaceChannelDrainageBatch`] couples a [`crate::coupling::SurfaceChannelBatch`] and a
//! [`crate::drainage::SwmmDrainage`] over a common interval `[start_s, end_s]`. Each trial runs
//! an outer midpoint-head fixed-point iteration on the surface↔drainage inlets while solving
//! the inner 2D surface↔1D channel/pool/structure exchange on the exact same interval.

mod batch;
mod capabilities;
mod error;
mod forcing_splice;
mod state;
mod trial;

pub use batch::SurfaceChannelDrainageBatch;
pub use capabilities::surface_channel_drainage_capabilities;
pub use error::SurfaceChannelDrainageError;
pub use state::{
    SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION, SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA,
    SurfaceChannelDrainageSetup, SurfaceChannelDrainageSnapshotData,
    SurfaceChannelDrainageStepDiagnostics,
};
pub use trial::SurfaceChannelDrainageTrial;
