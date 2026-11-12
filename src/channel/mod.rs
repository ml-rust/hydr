//! Declared channel geometry and reach flow for hydraulic applications.

mod flow;
mod section;

pub use flow::{
    ChannelBoundary, ChannelBoundaryConfig, ChannelBoundaryTrace, ChannelCellResult, ChannelConfig,
    ChannelError, ChannelForcingInterval, ChannelGeometry, ChannelInitialState,
    ChannelIntervalSnapshotData, ChannelLateralBinding, ChannelLateralDirection,
    ChannelLateralInput, ChannelLateralLedger, ChannelLateralReceipt, ChannelLateralStateData,
    ChannelReach, ChannelReachDefinition, ChannelReservoirTrace, ChannelRiemannDiagnostics,
    ChannelSnapshot, ChannelSnapshotData, ChannelStep, ChannelStepDiagnostics,
    ChannelTensorBoundary, ChannelTensorForcingInterval, ChannelTensorLateralInput,
    ChannelTensorWaterAmount, ChannelTriangularReconstruction, ChannelWaterLedger,
    RectangularChannelCell, TriangularChannelCell,
};
pub use section::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionError, CrossSectionPoint,
    CrossSectionProperties, CrossSectionQuery, CrossSectionRecoveredProperties,
    CrossSectionRecovery, CrossSectionRecoveryConfig, PRISMATIC_BAND_WORK_LIMIT,
    PrismaticAreaPointBandCoordinates, PrismaticAreaPointBandProperties,
    PrismaticAreaPointRecoveredProperties, PrismaticAreaPointRecovery, PrismaticPerimeterBatch,
    PrismaticPerimeterProperties, PrismaticPerimeterQuery, PrismaticPerimeterRecoveredProperties,
    PrismaticPerimeterRecovery, PrismaticRecoveredProperties, PrismaticRecoveryConfig,
    PrismaticSectionBatch, PrismaticSectionProperties, PrismaticSectionQuery,
    PrismaticSectionRecovery,
};

pub(crate) use flow::{ChannelCouplingForcing, ChannelCouplingTrial, ChannelReservoirOperands};

pub use flow::{
    PrismaticRiemannConfig, PrismaticRiemannError, PrismaticRiemannRejection,
    PrismaticRiemannResult, PrismaticRiemannWork,
};
