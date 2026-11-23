//! Per-barrel tensors built once for barrel geometry, inlet control and outlet control.

mod arcs;
mod certificates;
mod context;
mod floor;
mod frame;
mod inlet;
mod law;
mod outlet;
mod series;

pub(crate) use arcs::ArcRows;
pub(crate) use context::{Bounds, TensorBarrelContext};
pub(crate) use floor::FloorRows;
pub(crate) use frame::BarrelFrame;
pub(crate) use inlet::TensorInletContext;
pub use law::CulvertHint;
pub(crate) use law::{CulvertLaw, TensorCulvertLaw, TensorCulvertResult};
pub(crate) use outlet::TensorOutletContext;
pub(crate) use series::SeriesTensors;
