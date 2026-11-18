//! Native scalar endpoint traces, flux port faces and their physical fluxes.

mod algebra;
pub(in crate::channel::flow) mod assembly;
mod curves;
mod depth;
mod discharge;
mod evaluation;
#[cfg(test)]
mod fixture;
mod kernel;
mod port;
mod projection;
mod reservoir;
mod reservoir_head;
pub(in crate::channel::flow) mod types;
mod wave;

pub(in crate::channel::flow) use types::{
    DirectBoundaryFrame, EndpointInterior, NativeBoundaryTraces, PortFront,
};
