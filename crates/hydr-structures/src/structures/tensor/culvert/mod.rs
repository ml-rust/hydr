//! Per-barrel tensors built once for barrel geometry and inlet control.

mod context;
mod inlet;

pub(crate) use context::{Bounds, TensorBarrelContext};
pub(crate) use inlet::TensorInletContext;
