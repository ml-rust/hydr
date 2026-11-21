//! Per-barrel tensors built once for barrel geometry, inlet control and outlet control.

mod context;
mod inlet;
mod law;
mod outlet;

pub(crate) use context::{Bounds, TensorBarrelContext};
pub(crate) use inlet::TensorInletContext;
pub(crate) use law::{CulvertLaw, TensorCulvertLaw, TensorCulvertResult};
pub(crate) use outlet::TensorOutletContext;
