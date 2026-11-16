//! Declared channel geometry and reach flow for hydraulic applications.

mod flow;
mod section;

pub use flow::{
    ChannelBoundary, ChannelBoundaryConfig, ChannelBoundaryTrace, ChannelCellResult, ChannelConfig,
    ChannelError, ChannelForcingInterval, ChannelFrictionGeometryDiagnostics, ChannelGeometry,
    ChannelInitialState, ChannelIntervalSnapshotData, ChannelLateralBinding,
    ChannelLateralDirection, ChannelLateralInput, ChannelLateralLedger, ChannelLateralReceipt,
    ChannelLateralStateData, ChannelReach, ChannelReachDefinition, ChannelReservoirTrace,
    ChannelRiemannDiagnostics, ChannelSnapshot, ChannelSnapshotData, ChannelStep,
    ChannelStepDiagnostics, ChannelSurveyedConfig, ChannelTensorBoundary,
    ChannelTensorForcingInterval, ChannelTensorLateralInput, ChannelTensorWaterAmount,
    ChannelTriangularReconstruction, ChannelWaterLedger, RectangularChannelCell,
    SurveyedChannelCell, TriangularChannelCell,
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

pub(crate) use flow::{ChannelCouplingForcing, ChannelCouplingTrial, ChannelReservoirOperands};

pub use flow::{
    PrismaticPointRiemannResult, PrismaticPointTraceKind, PrismaticRiemannConfig,
    PrismaticRiemannError, PrismaticRiemannRejection, PrismaticRiemannResult, PrismaticRiemannWork,
};
