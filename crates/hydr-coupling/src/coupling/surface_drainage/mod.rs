//! Conservative coupling between 2D shallow-water surface flow and pinned EPA SWMM v5.2.4 drainage.

mod authority;
mod batch;
mod capabilities;
mod error;
pub(crate) mod exchange;
mod inlet;
mod state;

pub use authority::{DrainageControlAuthorityTable, DrainageControlClaim};
pub use batch::SurfaceDrainageBatch;
pub use capabilities::surface_drainage_capabilities;
pub use error::SurfaceDrainageError;
pub use exchange::SurfaceDrainageTrial;
pub use inlet::{
    SurfaceDrainageConfig, SurfaceDrainageInlet, SurfaceDrainageSetup, inlet_discharge_m3_per_s,
};
pub use state::{
    SURFACE_DRAINAGE_CONTRACT_VERSION, SURFACE_DRAINAGE_SNAPSHOT_SCHEMA,
    SurfaceDrainageInletTransfer, SurfaceDrainageSnapshotData, SurfaceDrainageStepDiagnostics,
};
