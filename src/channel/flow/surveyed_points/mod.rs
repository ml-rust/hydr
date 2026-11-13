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
mod paired_endpoint;
mod paired_residual;
mod paired_source_cache;
mod paired_source_invariant;
mod paired_sources;
mod pointed_depth_identity;
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
mod root_invocation;
mod shock_points;
mod source_binding;
pub(crate) mod source_velocity;
#[cfg(test)]
mod source_velocity_binding_tests;
pub(crate) mod types;
mod wave_curve;
#[cfg(test)]
mod wave_curve_tests;
mod wave_curve_types;
mod wet_depth;
mod wet_depth_mapping;
mod wet_depth_refinement;
mod wet_depth_selection;
mod wet_depth_types;
mod wet_root_charts;
mod wet_root_direct;
mod wet_root_search;
mod wet_root_seeds;
mod wet_root_state;
mod wet_root_types;
mod wet_roots;
mod wet_star;
mod wet_star_endpoint;
mod wet_star_types;
mod wet_wave_types;
mod wet_waves;
mod wet_waves_endpoint;
mod wet_waves_regions;
mod wet_waves_side;

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
