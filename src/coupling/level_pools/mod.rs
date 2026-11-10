//! Connected level pools with checked structure laws and optional explicit local sources.

mod batch;
mod context;
mod definition;
mod error;
mod exchange;
mod heads;
mod identity;
mod iteration;
mod mass;
mod output;
mod projection;
mod snapshot;
mod sources;
mod topology;
mod trial;

pub use batch::CoupledLevelPoolBatch;
pub use definition::{
    CoupledLevelPoolConfig, CoupledLevelPoolInitialState, CoupledLevelPoolSetup,
    CoupledPoolAttachment, CoupledPoolConnection, CoupledPoolSupport,
};
pub use error::CoupledLevelPoolError;
pub use output::{
    CoupledLevelPoolIterationData, CoupledLevelPoolStepDiagnostics, CoupledLevelPoolTransfer,
};
pub use snapshot::{CoupledLevelPoolSnapshot, CoupledLevelPoolSnapshotData};
pub use trial::CoupledLevelPoolTrial;

pub use sources::CoupledLevelPoolSourceInterval;
