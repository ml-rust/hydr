//! Standalone runoff storage primitives, independent of hydraulic attachment.

mod numerics;
mod reservoir;
mod subcatchment;
mod support;
mod tensor;

pub use support::{
    AdmittedInterval, BoundBoundaryInput, BoundaryBinding, CellRainAssignment, Contribution,
    ContributionKind, PartitionAssertion, RainSuppressionReceipt, SubcatchmentContributionInput,
    SupportAtom, SupportError, SupportErrorKind, SupportIntervalInput, SupportPlan,
    SupportPlanInput, SurfaceCellSupport, SurfaceObservationInput, WholeReplacement,
};

pub use reservoir::{
    RESERVOIR_CONTRACT_VERSION, RESERVOIR_SNAPSHOT_SCHEMA, ReservoirBatch, ReservoirConfig,
    ReservoirElement, ReservoirError, ReservoirInput, ReservoirLedger, ReservoirOutput,
    ReservoirParameters, ReservoirReceipt, ReservoirSnapshot, ReservoirSnapshotData,
    ReservoirState, ReservoirStepDiagnostics, ReservoirTrial, reservoir_capabilities,
};

pub use subcatchment::{
    SUBCATCHMENT_CONTRACT_VERSION, SUBCATCHMENT_SNAPSHOT_SCHEMA, Subarea, SubareaOutput,
    SubareaReceipt, SubcatchmentBatch, SubcatchmentBridgeReceipt, SubcatchmentConfig,
    SubcatchmentDefinition, SubcatchmentDrainageReceipt, SubcatchmentError, SubcatchmentGeometry,
    SubcatchmentInput, SubcatchmentLedger, SubcatchmentOutput, SubcatchmentParameters,
    SubcatchmentPartitionReceipt, SubcatchmentPhaseDiagnostics, SubcatchmentReceipt,
    SubcatchmentRouting, SubcatchmentSnapshot, SubcatchmentSnapshotData, SubcatchmentState,
    SubcatchmentStepDiagnostics, SubcatchmentTransfer, SubcatchmentTrial,
    subcatchment_capabilities,
};
