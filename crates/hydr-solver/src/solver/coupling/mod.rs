//! Hidden coordinator API for surface exchange with a coupled component.

mod operands;
mod port_ghosts;
mod ports;
mod stage;

#[cfg(test)]
pub(in crate::solver) mod fixture;

pub use super::tensor::port_faces::PortFluxes;
pub use operands::{ExchangeSource, SurfaceLinks};
pub use port_ghosts::PortGhosts;
pub use ports::{PortReachEnd, SurfacePortSpec, SurfacePorts};
pub use stage::{CouplingCompletion, CouplingPorts, CouplingPreparation, SurfaceCouplingStage};
