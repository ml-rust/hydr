//! Grid, model, forcing and shared numerics for `hydr`.
//!
//! This crate implements part of `hydr`. Users depend on `hydr`, which re-exports the public modules.

pub mod error;
pub mod forcing;
pub mod grid;
pub mod model;

#[doc(hidden)]
pub mod contract;
#[doc(hidden)]
pub mod numerics;
#[doc(hidden)]
pub mod tensor;
