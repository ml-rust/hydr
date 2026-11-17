//! Boundary, forcing, config and error contract shared by the solver and numerics.

mod boundary;
mod config;
mod error;
mod forcing;
mod identity;

pub use boundary::exterior_cell;
pub use boundary::{
    Axis, BoundaryConditions, BoundaryFace, BoundaryForcing, DischargeBoundary, ExteriorBoundary,
    ExteriorState, FluxPort, StageBoundary,
};
pub use config::{BoundaryTolerances, SolverConfig};
pub use error::EngineError;
pub use forcing::{ForcingInterval, ForcingSchedule};
pub use identity::package_identity;
