//! Advance channel reaches and finite level pools through common, conservatively paired intervals.

mod accounting;
mod accounting_lateral;
mod accounting_math;
mod accounting_physical;
mod batch;
mod boundary_history;
mod capabilities;
mod context;
mod context_types;
mod counters;
mod definition;
mod donors;
mod error;
mod exchange;
mod forcing;
mod forcing_definition;
mod identity;
mod iteration;
mod output;
mod packets;
mod projection;
mod reductions;
mod restoration;
mod snapshot;
mod snapshot_values;
mod state;
mod topology;
mod trial;

pub use batch::ChannelPoolBatch;
#[cfg(test)]
pub(in crate::coupling) use batch::tests::definition as three_band_reach;
pub use capabilities::channel_pool_capabilities;
pub use definition::{
    ChannelPoolConfig, ChannelPoolConnection, ChannelPoolEndpoint, ChannelPoolInitialState,
    ChannelPoolReach, ChannelPoolSetup,
};
pub use error::ChannelPoolError;
pub use forcing_definition::{
    ChannelPoolForcingInterval, ChannelPoolReachForcing, ChannelPoolTensorForcingInterval,
    ChannelPoolTensorReachForcing,
};
pub use output::{
    ChannelPoolConnectionLedger, ChannelPoolDonorCut, ChannelPoolIterationData,
    ChannelPoolStepDiagnostics, ChannelPoolTransfer, ChannelPoolWaterBalance,
};
pub use snapshot::{
    CHANNEL_POOL_CONTRACT_VERSION, CHANNEL_POOL_SNAPSHOT_SCHEMA, ChannelPoolCounterData,
    ChannelPoolIntervalAccountingData, ChannelPoolIntervalSnapshotData,
    ChannelPoolNativeIntervalData, ChannelPoolPreviousData, ChannelPoolSnapshot,
    ChannelPoolSnapshotData,
};
pub use trial::ChannelPoolTrial;
