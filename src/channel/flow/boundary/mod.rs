//! Native scalar endpoint traces and physical fluxes.

mod algebra;
pub(in crate::channel::flow) mod assembly;
mod curves;
mod discharge;
mod evaluation;
mod kernel;
mod projection;
pub(in crate::channel::flow) mod types;
mod wave;

pub(in crate::channel::flow) use types::{
    EndpointInterior, NativeBoundaryTraces, ScalarBoundaryFrame,
};
