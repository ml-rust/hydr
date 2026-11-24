//! Runtime-selected shallow-water execution and accepted-state contract.

mod accounting;
mod admission;
mod advance;
mod capabilities;
mod coastal;
mod compatibility;
mod continuation;
pub(crate) mod coupling;
mod definition;
mod engine;
mod f32_profile;
mod forcing_history;
mod geometry;
mod output;
mod receipts;
mod restore;
mod setup;
mod snapshot;
mod stepping;
mod subgrid;
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
pub use capabilities::{CONTRACT_VERSION, surface_solver_capabilities};
pub use coastal::CoastalForcing;
pub use engine::Solver;
pub use f32_profile::{
    SURFACE_SOLVER_F32_CONTRACT_VERSION, SURFACE_SOLVER_F32_SNAPSHOT_SCHEMA, SurfaceF32Boundaries,
    SurfaceF32CoastalForcing, SurfaceF32SideBoundary, SurfaceF32Soil, SurfaceF32SpatialOrder,
    SurfaceSolverF32, SurfaceSolverF32Config, SurfaceSolverF32Ledger, SurfaceSolverF32SnapshotData,
    surface_solver_f32_capabilities,
};
pub use forcing_history::ForcingPrefix;
pub use geometry::{
    SurfaceBedEpoch, SurfaceBedSchedule, SurfaceBedUpdate, active_bed_at,
    active_surface_bed_epochs, bed_digest_bits, check_surface_bed_schedules,
    next_surface_bed_event_after,
};
pub use output::{CellResult, FaceTransfer};
pub use setup::{InitialState, SolverSetup};
pub use snapshot::{SNAPSHOT_SCHEMA, Snapshot, SnapshotData};
pub use subgrid::{SubgridCellCurve, SubgridTerrain};
