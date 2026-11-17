//! Native Modified Green–Ampt state transitions, hydraulic integration and tensor storage.

mod integration;
pub(crate) mod roundoff;
mod storage;
pub(crate) mod transition;

pub(crate) use storage::{TensorSoilContext, TensorSoilState};
pub(crate) use transition::advance;
