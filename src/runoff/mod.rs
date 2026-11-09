//! Standalone runoff storage primitives, independent of hydraulic attachment.

mod reservoir;

pub use reservoir::{
    ReservoirBatch, ReservoirConfig, ReservoirElement, ReservoirError, ReservoirInput,
    ReservoirLedger, ReservoirOutput, ReservoirParameters, ReservoirReceipt, ReservoirSnapshot,
    ReservoirSnapshotData, ReservoirState, ReservoirStepDiagnostics, ReservoirTrial,
};
