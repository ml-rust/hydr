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
#[cfg(test)]
mod corpus_coverage;
mod dry_vacuum;
mod dry_vacuum_candidates;
mod dry_vacuum_evidence;
mod dry_vacuum_positions;
mod dry_vacuum_types;
mod paired_endpoint;
mod paired_residual;
mod paired_source_cache;
mod paired_source_invariant;
mod paired_source_panels;
mod paired_sources;
mod point_candidate_types;
mod point_candidates;
mod point_candidates_admission;
mod point_candidates_demand;
mod point_candidates_evidence;
mod point_candidates_source;
mod point_flux;
mod point_flux_admission;
mod point_flux_arithmetic;
mod point_flux_assembly;
mod point_flux_association;
mod point_flux_operands;
mod point_flux_pressure;
mod point_flux_projection;
mod point_flux_quantum;
mod point_flux_selection;
mod point_flux_support;
mod point_flux_support_metrics;
mod point_flux_types;
mod point_root_depth;
mod point_root_session;
mod point_root_wet;
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
mod source_sonic;
mod source_sonic_binding;
mod source_sonic_callback;
mod source_sonic_certificates;
mod source_sonic_charts;
mod source_sonic_invariant;
mod source_sonic_numerical;
mod source_sonic_policy;
mod source_sonic_quadrature;
mod source_sonic_quadrature_capacity;
mod source_sonic_quadrature_request;
mod source_sonic_quadrature_types;
mod source_sonic_refinement;
mod source_sonic_residual;
mod source_sonic_selected_fields;
mod source_sonic_selection;
mod source_sonic_start;
mod source_sonic_types;
pub(crate) mod source_velocity;
#[cfg(test)]
mod source_velocity_binding_tests;
pub(crate) mod types;
mod wave_curve;
#[cfg(test)]
mod wave_curve_tests;
mod wave_curve_types;
mod wet_candidates;
mod wet_candidates_fans;
mod wet_candidates_star;
mod wet_depth;
mod wet_depth_mapping;
mod wet_depth_refinement;
mod wet_depth_selection;
mod wet_depth_types;
mod wet_fan_membership;
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
