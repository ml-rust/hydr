//! Conservative water exchange between hydraulic components.

mod checkpoint;

pub mod channel_pools;
#[doc(no_inline)]
pub use channel_pools::{
    CHANNEL_POOL_CONTRACT_VERSION, CHANNEL_POOL_SNAPSHOT_SCHEMA, ChannelPoolBatch,
    ChannelPoolConfig, ChannelPoolConnection, ChannelPoolConnectionLedger, ChannelPoolCounterData,
    ChannelPoolDonorCut, ChannelPoolEndpoint, ChannelPoolError, ChannelPoolForcingInterval,
    ChannelPoolInitialState, ChannelPoolIntervalAccountingData, ChannelPoolIntervalSnapshotData,
    ChannelPoolIterationData, ChannelPoolNativeIntervalData, ChannelPoolPreviousData,
    ChannelPoolReach, ChannelPoolReachForcing, ChannelPoolSetup, ChannelPoolSnapshot,
    ChannelPoolSnapshotData, ChannelPoolStepDiagnostics, ChannelPoolTensorForcingInterval,
    ChannelPoolTensorReachForcing, ChannelPoolTransfer, ChannelPoolTrial, ChannelPoolWaterBalance,
    channel_pool_capabilities,
};

pub mod level_pools;

#[doc(no_inline)]
pub use level_pools::CoupledPoolInflowValue;
#[doc(no_inline)]
pub use level_pools::{
    COUPLED_LEVEL_POOL_CONTRACT_VERSION, COUPLED_LEVEL_POOL_SNAPSHOT_SCHEMA, CoupledLevelPoolBatch,
    CoupledLevelPoolConfig, CoupledLevelPoolError, CoupledLevelPoolForcingInterval,
    CoupledLevelPoolInitialState, CoupledLevelPoolIterationData, CoupledLevelPoolSetup,
    CoupledLevelPoolSnapshot, CoupledLevelPoolSnapshotData, CoupledLevelPoolStartData,
    CoupledLevelPoolStepDiagnostics, CoupledLevelPoolTransfer, CoupledLevelPoolTrial,
    CoupledPoolAttachment, CoupledPoolConnection, CoupledPoolInflow, CoupledPoolInflowBinding,
    CoupledPoolInflowLedger, CoupledPoolInflowReceipt, CoupledPoolSupport,
    coupled_level_pool_capabilities,
};

pub mod runoff_pools;
#[doc(no_inline)]
pub use runoff_pools::{
    RUNOFF_POOL_CONTRACT_VERSION, RUNOFF_POOL_SNAPSHOT_SCHEMA, RunoffPoolBatch, RunoffPoolConfig,
    RunoffPoolError, RunoffPoolForcingInterval, RunoffPoolInitialState, RunoffPoolMeteorology,
    RunoffPoolOutletBinding, RunoffPoolSetup, RunoffPoolSnapshot, RunoffPoolSnapshotData,
    RunoffPoolStepDiagnostics, RunoffPoolTrial, RunoffPoolWaterBalance, runoff_pool_capabilities,
};

pub mod surface_channels;
#[doc(no_inline)]
pub use surface_channels::{
    SURFACE_CHANNEL_CONTRACT_VERSION, SURFACE_CHANNEL_SNAPSHOT_SCHEMA,
    SURFACE_CHANNEL_TANGENT_TOLERANCE, SurfaceChannelBatch, SurfaceChannelConfig,
    SurfaceChannelCounterData, SurfaceChannelDefinition, SurfaceChannelError,
    SurfaceChannelForcingInterval, SurfaceChannelInitialState, SurfaceChannelIntervalSnapshotData,
    SurfaceChannelLink, SurfaceChannelLinkLedger, SurfaceChannelLinkTransfer,
    SurfaceChannelPreviousData, SurfaceChannelReach, SurfaceChannelReachForcing,
    SurfaceChannelSetup, SurfaceChannelSetupData, SurfaceChannelSnapshot,
    SurfaceChannelSnapshotData, SurfaceChannelStepDiagnostics, SurfaceChannelTrial,
    SurfaceChannelWaterBalance, SurfaceChannelWeirRegime,
};
