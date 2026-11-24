//! Opt-in `F32` 2D shallow-water surface solver for CPU, CUDA, and WebGPU (`WgpuRuntime`).
//!
//! Runs all tensor computations natively in `DType::F32` without `F64` or `CorrectlyRoundedOps`,
//! while keeping the certified `F64` [`super::Solver`] unchanged.

mod capabilities;
mod engine;
mod flux;
mod state;

pub use capabilities::surface_solver_f32_capabilities;
pub use engine::SurfaceSolverF32;
pub use state::{
    SURFACE_SOLVER_F32_CONTRACT_VERSION, SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA,
    SurfaceSolverF32Config, SurfaceSolverF32Ledger, SurfaceSolverF32SnapshotData,
};
