//! Persistent rectangular-gate geometry, explicit openings, and evaluated witnesses.

mod context;
mod input;
mod proposal;

pub(crate) use context::TensorGateContext;
pub(crate) use input::TensorGateInput;
pub(crate) use proposal::{TensorGateGeometry, TensorGateResult};

pub(crate) mod trajectory;
