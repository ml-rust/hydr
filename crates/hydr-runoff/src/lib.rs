//! Runoff generation for `hydr`.
//!
//! This crate implements part of `hydr`. Users depend on `hydr`, which re-exports the public modules.

use hydr_core::{contract, model, numerics, tensor};
use hydr_soil::soil;

pub mod runoff;
