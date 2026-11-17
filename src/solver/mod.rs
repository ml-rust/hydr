//! Runtime-selected shallow-water execution and accepted-state contract.

mod accounting;
mod admission;
mod advance;
mod boundary;
mod capabilities;
mod compatibility;
mod config;
mod definition;
mod engine;
mod error;
mod forcing;
mod forcing_history;
mod output;
mod receipts;
mod restore;
mod setup;
mod snapshot;
mod stepping;

pub use accounting::{
    BoundaryTraceDiagnostics, SoilSourceReceipt, SourceReceipt, StepDiagnostics,
    TransmissiveDiagnostics, WaterLedger,
};
pub use advance::{AdvanceReport, AdvanceStatus};
pub use boundary::{
    Axis, BoundaryConditions, BoundaryFace, BoundaryForcing, DischargeBoundary, ExteriorBoundary,
    ExteriorState, FluxPort, StageBoundary,
};
pub use capabilities::{CONTRACT_VERSION, Capabilities, capabilities};
pub use config::{BoundaryTolerances, SolverConfig};
pub use engine::Solver;
pub use error::EngineError;
pub use forcing::{ForcingInterval, ForcingSchedule};
pub use forcing_history::ForcingPrefix;
pub use output::{CellResult, FaceTransfer};
pub use setup::{InitialState, SolverSetup};
pub use snapshot::{SNAPSHOT_SCHEMA, Snapshot, SnapshotData};
