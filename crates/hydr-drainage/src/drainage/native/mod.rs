//! Pure-Rust 1D dynamic-wave Preissmann-slot pressurized-pipe drainage network solver.
//!
//! Runs with `#![forbid(unsafe_code)]` and zero external binary dependencies, supporting
//! free-surface gravity flow, smooth pressurized surcharge transitions via the Preissmann slot,
//! link control settings, exact SI continuity accounting, transactional rollback, and portable
//! checkpoints.

mod capabilities;
mod engine;
mod geometry;
mod network;

pub use capabilities::native_pipe_drainage_capabilities;
pub use engine::{NativePipeNetwork, NativePipeTrial};
pub use geometry::{ConduitHydraulics, PipeCrossSection};
pub use network::{
    NATIVE_PIPE_CONTRACT_VERSION, NATIVE_PIPE_SNAPSHOT_SCHEMA, NativePipeConduitState,
    NativePipeConfig, NativePipeLedger, NativePipeNetworkSpec, NativePipeNodeState,
    NativePipeSnapshotData, NativePipeState, PipeConduitSpec, PipeNodeKind, PipeNodeSpec,
    PipeOutfallBoundary,
};
