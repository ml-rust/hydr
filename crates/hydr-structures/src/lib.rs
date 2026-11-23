//! Hydraulic structures for `hydr`.
//!
//! This crate implements part of `hydr`. Users depend on `hydr`, which re-exports the public modules.

use hydr_core::{numerics, tensor};

pub mod structures;

#[doc(hidden)]
pub mod internal {
    pub use crate::structures::blockage::effective::culvert_bed_elevations;
    pub use crate::structures::coupling::CouplingStructures;
    pub use crate::structures::identity::{bits, same_elements};
    pub use crate::structures::tensor::TensorStructureResult;
    pub use crate::structures::tensor::culvert::CulvertHint;
}
