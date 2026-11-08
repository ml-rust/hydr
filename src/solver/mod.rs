//! Sequential CPU shallow-water reference and accepted-state execution contract.

mod accounting;
mod boundary;
mod capabilities;
mod compatibility;
mod config;
mod engine;
mod error;
mod forcing;
mod output;
mod snapshot;
mod stepping;

pub use accounting::{StepDiagnostics, WaterLedger};
pub use boundary::{Axis, BoundaryConditions, BoundaryFace, FluxPort};
pub use capabilities::{CONTRACT_VERSION, Capabilities, capabilities};
pub use config::SolverConfig;
pub use engine::{AdvanceReport, AdvanceStatus, Solver};
pub use error::EngineError;
pub use forcing::{ForcingInterval, ForcingSchedule};
pub use output::{CellResult, FaceTransfer};
pub use snapshot::{SNAPSHOT_SCHEMA, Snapshot, SnapshotData};
