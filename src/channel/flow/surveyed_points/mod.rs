//! Exact conserved-point sources for surveyed Riemann calculations.

pub(crate) mod analytic_bottom;
mod analytic_bottom_arithmetic;
mod analytic_endpoint;
#[cfg(test)]
mod analytic_endpoint_tests;
mod chart_admission;
mod chart_brackets;
mod chart_root_callback;
mod chart_root_derivative;
mod chart_roots;
#[cfg(test)]
mod chart_test_support;
mod paired_residual;
mod paired_source_cache;
mod paired_source_invariant;
mod paired_sources;
mod potential_absolute;
mod potential_analytic;
#[cfg(test)]
mod potential_band_retention_tests;
pub(crate) mod potential_callbacks;
mod potential_increment;
#[cfg(test)]
mod potential_lanes;
mod potential_numerical;
mod potential_range;
#[cfg(test)]
mod potential_range_tests;
mod potential_sonic;
#[cfg(test)]
mod potential_underflow_tests;
mod shock_points;
pub(crate) mod source_velocity;
#[cfg(test)]
mod source_velocity_binding_tests;
pub(crate) mod types;
mod wave_curve;
#[cfg(test)]
mod wave_curve_tests;
mod wave_curve_types;

#[cfg(test)]
#[path = "source_tests.rs"]
mod source_tests;

#[cfg(test)]
#[path = "analytic_bottom_tests.rs"]
mod analytic_bottom_tests;

#[cfg(test)]
mod potential_analytic_tests;

#[cfg(test)]
mod potential_sonic_tests;
