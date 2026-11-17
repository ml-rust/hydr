//! Connected level pools with checked laws, explicit local sources, and prescribed external inflows.

mod batch;
mod capabilities;
mod context;
mod definition;
mod error;
mod exchange;
mod forcing;
mod heads;
mod identity;
mod inflows;
mod iteration;
mod mass;
mod native_accounting;
mod output;
mod projection;
mod restoration;
mod snapshot;
mod topology;
mod trial;

pub use batch::CoupledLevelPoolBatch;
pub use capabilities::coupled_level_pool_capabilities;
pub use definition::{
    CoupledLevelPoolConfig, CoupledLevelPoolInitialState, CoupledLevelPoolSetup,
    CoupledPoolAttachment, CoupledPoolConnection, CoupledPoolSupport,
};
pub use error::CoupledLevelPoolError;
pub use output::{
    CoupledLevelPoolIterationData, CoupledLevelPoolStepDiagnostics, CoupledLevelPoolTransfer,
};
pub use snapshot::{
    COUPLED_LEVEL_POOL_CONTRACT_VERSION, COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA,
    CoupledLevelPoolSnapshot, CoupledLevelPoolSnapshotData, CoupledLevelPoolStartData,
};
pub use trial::CoupledLevelPoolTrial;

pub use forcing::CoupledLevelPoolForcingInterval;

pub use inflows::{
    CoupledPoolInflow, CoupledPoolInflowBinding, CoupledPoolInflowLedger, CoupledPoolInflowReceipt,
    CoupledPoolInflowValue,
};
