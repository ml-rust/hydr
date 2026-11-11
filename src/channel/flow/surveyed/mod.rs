//! Certified interface fluxes for admitted surveyed prismatic sections.

pub(crate) mod acceptance;
pub(crate) mod band_lanes;
pub(crate) mod config;
pub(crate) mod controller;
pub(crate) mod curves;
pub(crate) mod derivatives;
pub(crate) mod equation_ops;
pub(crate) mod error;
pub(crate) mod integration;
pub(crate) mod output;
pub(crate) mod pressure;
pub(crate) mod root_calls;
pub(crate) mod roots;
pub(crate) mod sample_candidates;
pub(crate) mod sample_fans;
pub(crate) mod sample_merge;
pub(crate) mod sample_waves;
pub(crate) mod sampling;
pub(crate) mod types;
pub(crate) mod work;

pub use config::PrismaticRiemannConfig;
pub use error::{PrismaticRiemannError, PrismaticRiemannRejection};
pub use output::PrismaticRiemannResult;
pub use work::PrismaticRiemannWork;

#[cfg(test)]
mod derivative_data;
#[cfg(test)]
mod derivative_tests;
#[cfg(test)]
mod error_tests;

#[cfg(test)]
mod controller_resolution_tests;

pub(crate) mod analytic_bottom;
#[cfg(test)]
mod analytic_bottom_tests;
