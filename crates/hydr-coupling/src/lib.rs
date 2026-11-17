//! Pool coupling for `hydr`.
//!
//! This crate implements part of `hydr`. Users depend on `hydr`, which re-exports the public modules.

use hydr_channel::channel;
use hydr_core::{forcing, numerics};
use hydr_runoff::runoff;
use hydr_storage::storage;
use hydr_structures::structures;

pub mod coupling;
