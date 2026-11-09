//! Persistent rectangular-gate geometry, explicit openings, and evaluated witnesses.

mod context;
mod input;
mod proposal;

pub(crate) use context::CpuGateContext;
pub(crate) use input::CpuGateInput;
pub(crate) use proposal::{CpuGateGeometry, CpuGateResult};
