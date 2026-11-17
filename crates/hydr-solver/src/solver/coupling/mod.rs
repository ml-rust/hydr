//! Hidden coordinator API for surface exchange with a coupled component.

mod operands;
mod stage;

#[cfg(test)]
pub(in crate::solver) mod fixture;

pub use operands::{ExchangeSource, SurfaceLinks};
pub use stage::{CouplingCompletion, CouplingPreparation, SurfaceCouplingStage};
