//! Cached 2D cell tensors and face execution for the solver.

mod boundary;
mod boundary_trace;
mod exterior;
pub(crate) mod land;
pub(crate) mod land_sources;
mod storage;
mod topology;
pub(crate) mod transport;

pub(crate) use storage::{TensorDomain, TensorSources, TensorState};
pub(crate) use topology::TensorTopology;
