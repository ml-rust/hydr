//! Shallow-water solver for `hydr`.
//!
//! This crate implements part of `hydr`. Users depend on `hydr`, which re-exports the public modules.

#[cfg(test)]
use hydr_core::grid;
use hydr_core::{contract, model, numerics, tensor};
use hydr_soil::soil;

pub mod solver;

/// Coordinator-only surface exchange API. Not part of the public contract.
#[doc(hidden)]
pub mod internal {
    pub use crate::solver::coupling::{
        CouplingCompletion, CouplingPreparation, ExchangeSource, SurfaceCouplingStage, SurfaceLinks,
    };
}
