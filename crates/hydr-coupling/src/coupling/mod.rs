//! Conservative water exchange between hydraulic components.

pub mod channel_pools;
#[doc(no_inline)]
pub use channel_pools::{
    ChannelPoolBatch, ChannelPoolConfig, ChannelPoolConnection, ChannelPoolConnectionLedger,
    ChannelPoolCounterData, ChannelPoolDonorCut, ChannelPoolEndpoint, ChannelPoolError,
    ChannelPoolForcingInterval, ChannelPoolInitialState, ChannelPoolIntervalAccountingData,
    ChannelPoolIntervalSnapshotData, ChannelPoolIterationData, ChannelPoolNativeIntervalData,
    ChannelPoolPreviousData, ChannelPoolReach, ChannelPoolReachForcing, ChannelPoolSetup,
    ChannelPoolSnapshot, ChannelPoolSnapshotData, ChannelPoolStepDiagnostics,
    ChannelPoolTensorForcingInterval, ChannelPoolTensorReachForcing, ChannelPoolTransfer,
    ChannelPoolTrial, ChannelPoolWaterBalance,
};

pub mod level_pools;

#[doc(no_inline)]
pub use level_pools::CoupledPoolInflowValue;
#[doc(no_inline)]
pub use level_pools::{
    CoupledLevelPoolBatch, CoupledLevelPoolConfig, CoupledLevelPoolError,
    CoupledLevelPoolForcingInterval, CoupledLevelPoolInitialState, CoupledLevelPoolIterationData,
    CoupledLevelPoolSetup, CoupledLevelPoolSnapshot, CoupledLevelPoolSnapshotData,
    CoupledLevelPoolStartData, CoupledLevelPoolStepDiagnostics, CoupledLevelPoolTransfer,
    CoupledLevelPoolTrial, CoupledPoolAttachment, CoupledPoolConnection, CoupledPoolInflow,
    CoupledPoolInflowBinding, CoupledPoolInflowLedger, CoupledPoolInflowReceipt,
    CoupledPoolSupport,
};

pub mod runoff_pools;
#[doc(no_inline)]
pub use runoff_pools::{
    RunoffPoolBatch, RunoffPoolConfig, RunoffPoolError, RunoffPoolForcingInterval,
    RunoffPoolInitialState, RunoffPoolMeteorology, RunoffPoolOutletBinding, RunoffPoolSetup,
    RunoffPoolSnapshot, RunoffPoolSnapshotData, RunoffPoolStepDiagnostics, RunoffPoolTrial,
    RunoffPoolWaterBalance,
};
