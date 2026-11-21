//! Certified interface fluxes for admitted surveyed prismatic sections.

pub(crate) mod acceptance;
pub(crate) mod band_lanes;
pub(crate) mod config;
pub(crate) mod controller;
mod curve_evaluation;
pub(crate) mod curves;
mod depth_potential;
pub(crate) mod derivatives;
pub(crate) mod equation_ops;
pub(crate) mod error;
pub(crate) mod integration;
mod lane_callbacks;
pub(crate) mod lane_quadrature;
pub(crate) mod lane_split;
pub(crate) mod output;
pub(crate) mod potential_difference;
pub(crate) mod potential_difference_derivatives;
pub(crate) mod potential_difference_lanes;
pub(crate) mod pressure;
pub(crate) mod pressure_segments;
mod refinement;
mod reflection;
pub(crate) mod root_calls;
mod root_evaluation;
pub(crate) mod roots;
pub(crate) mod sample_candidates;
pub(crate) mod sample_fans;
pub(crate) mod sample_merge;
pub(crate) mod sample_waves;
pub(crate) mod sampling;
mod shock;
#[cfg(test)]
mod shock_case_data;
mod star_bracket;
pub(crate) mod types;
pub(crate) mod work;

pub use config::PrismaticRiemannConfig;
pub use error::{PrismaticRiemannError, PrismaticRiemannRejection};
pub use output::PrismaticRiemannResult;
pub use work::PrismaticRiemannWork;

#[cfg(test)]
mod derivative_data;

pub(crate) mod analytic_bottom;

#[cfg(test)]
pub(crate) mod potential_difference_test_data;
#[cfg(test)]
pub(crate) mod potential_difference_test_support;
