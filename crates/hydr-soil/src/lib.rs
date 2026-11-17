//! Modified Green-Ampt soil infiltration for `hydr`.
//!
//! This crate implements part of `hydr`. Users depend on `hydr`, which re-exports the public modules.

use hydr_core::{model, numerics, tensor};

pub mod soil;

#[doc(hidden)]
pub mod internal {
    pub use crate::soil::kernel::TensorSoilContext;
    pub use crate::soil::{SoilReceiptInput, SoilSurfaceTrial, check_source_receipt};
}
