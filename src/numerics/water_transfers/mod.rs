//! Deterministic physical transfer assembly and absolute rounding bounds.

mod error;
mod rounding;
mod tree;

#[cfg(test)]
mod tests;

pub(crate) use error::WaterTransferError;
pub(crate) use rounding::{finite, reject, upward_add};
pub(crate) use tree::{AggregationKind, TransferTree};
