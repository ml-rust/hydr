//! Per-barrel tensors built once for barrel geometry, inlet control and outlet control.

mod arcs;
mod context;
mod inlet;
mod law;
mod outlet;

pub(crate) use arcs::ArcRows;
pub(crate) use context::{Bounds, SeriesTensors, TensorBarrelContext, reject_arc_barrels};
pub(crate) use inlet::TensorInletContext;
pub use law::CulvertHint;
pub(crate) use law::{CulvertLaw, TensorCulvertLaw, TensorCulvertResult};
pub(crate) use outlet::TensorOutletContext;
