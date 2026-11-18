//! Deterministic physical transfer assembly and absolute rounding bounds.

mod error;
mod rounding;
mod tree;

pub use error::WaterTransferError;
pub use rounding::{finite, reject, upward_add};
pub use tree::{AggregationKind, TransferTree};
