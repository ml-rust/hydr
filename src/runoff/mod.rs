//! Standalone runoff storage primitives, independent of hydraulic attachment.

mod reservoir;
mod subcatchment;

pub use reservoir::{
    ReservoirBatch, ReservoirConfig, ReservoirElement, ReservoirError, ReservoirInput,
    ReservoirLedger, ReservoirOutput, ReservoirParameters, ReservoirReceipt, ReservoirSnapshot,
    ReservoirSnapshotData, ReservoirState, ReservoirStepDiagnostics, ReservoirTrial,
};

pub use subcatchment::{
    Subarea, SubareaOutput, SubareaReceipt, SubcatchmentBatch, SubcatchmentBridgeReceipt,
    SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentDrainageReceipt, SubcatchmentError,
    SubcatchmentGeometry, SubcatchmentInput, SubcatchmentLedger, SubcatchmentOutput,
    SubcatchmentParameters, SubcatchmentPartitionReceipt, SubcatchmentPhaseDiagnostics,
    SubcatchmentReceipt, SubcatchmentRouting, SubcatchmentSnapshot, SubcatchmentSnapshotData,
    SubcatchmentState, SubcatchmentStepDiagnostics, SubcatchmentTransfer, SubcatchmentTrial,
};
