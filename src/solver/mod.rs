//! Runtime-selected shallow-water execution and accepted-state contract.

mod accounting;
mod admission;
mod advance;
mod capabilities;
mod compatibility;
mod continuation;
mod definition;
mod engine;
mod forcing_history;
mod output;
mod receipts;
mod restore;
mod setup;
mod snapshot;
mod stepping;
mod tensor;

pub use crate::contract::{
    Axis, BoundaryConditions, BoundaryFace, BoundaryForcing, BoundaryTolerances, DischargeBoundary,
    EngineError, ExteriorBoundary, ExteriorState, FluxPort, ForcingInterval, ForcingSchedule,
    SolverConfig, StageBoundary,
};
pub use accounting::{
    BoundaryTraceDiagnostics, SoilSourceReceipt, SourceReceipt, StepDiagnostics,
    TransmissiveDiagnostics, WaterLedger,
};
pub use advance::{AdvanceReport, AdvanceStatus};
pub use capabilities::{CONTRACT_VERSION, Capabilities, capabilities};
pub use engine::Solver;
pub use forcing_history::ForcingPrefix;
pub use output::{CellResult, FaceTransfer};
pub use setup::{InitialState, SolverSetup};
pub use snapshot::{SNAPSHOT_SCHEMA, Snapshot, SnapshotData};
