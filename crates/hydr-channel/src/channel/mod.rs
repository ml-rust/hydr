//! Declared channel geometry and reach flow for hydraulic applications.

pub(crate) mod flow;
mod section;

pub use flow::{
    CHANNEL_CONTRACT_VERSION, CHANNEL_SNAPSHOT_SCHEMA, ChannelBoundary, ChannelBoundaryConfig,
    ChannelBoundaryTrace, ChannelCellResult, ChannelConfig, ChannelEnd, ChannelError,
    ChannelFluxPort, ChannelForcingInterval, ChannelFrictionGeometryDiagnostics, ChannelGeometry,
    ChannelInitialState, ChannelIntervalSnapshotData, ChannelLateralBinding,
    ChannelLateralDirection, ChannelLateralInput, ChannelLateralLedger, ChannelLateralReceipt,
    ChannelLateralStateData, ChannelReach, ChannelReachDefinition, ChannelReservoirTrace,
    ChannelRiemannDiagnostics, ChannelSnapshot, ChannelSnapshotData, ChannelStep,
    ChannelStepDiagnostics, ChannelSurveyedConfig, ChannelTensorBoundary, ChannelTensorFluxPort,
    ChannelTensorForcingInterval, ChannelTensorLateralInput, ChannelTensorWaterAmount,
    ChannelTriangularReconstruction, ChannelWaterLedger, RectangularChannelCell,
    SurveyedChannelCell, TriangularChannelCell, rectangular_reach_capabilities,
    surveyed_reach_capabilities, triangular_reach_capabilities,
};
pub use section::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionError, CrossSectionPoint,
    CrossSectionProperties, CrossSectionQuery, CrossSectionRecoveredProperties,
    CrossSectionRecovery, CrossSectionRecoveryConfig, PRISMATIC_BAND_WORK_LIMIT,
    PrismaticAreaPointBandCoordinates, PrismaticAreaPointBandProperties,
    PrismaticAreaPointRecoveredProperties, PrismaticAreaPointRecovery,
    PrismaticPerimeterAreaPointRecoveredProperties, PrismaticPerimeterAreaPointRecovery,
    PrismaticPerimeterBatch, PrismaticPerimeterProperties, PrismaticPerimeterQuery,
    PrismaticPerimeterRecoveredProperties, PrismaticPerimeterRecovery,
    PrismaticRecoveredProperties, PrismaticRecoveryConfig, PrismaticSectionBatch,
    PrismaticSectionProperties, PrismaticSectionQuery, PrismaticSectionRecovery,
};

pub use flow::{
    PrismaticPointRiemannResult, PrismaticPointTraceKind, PrismaticRiemannConfig,
    PrismaticRiemannError, PrismaticRiemannRejection, PrismaticRiemannResult, PrismaticRiemannWork,
};
