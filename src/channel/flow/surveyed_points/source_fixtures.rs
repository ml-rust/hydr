#[path = "source_case_index.rs"]
mod case_index;
#[path = "source_fixture_model.rs"]
mod model;
#[path = "source_profiles.rs"]
mod profiles;
#[path = "source_cases_three_bands_wide_bottom_dry_fan.rs"]
mod source_cases_planned_three_bands_dry_planned_three_bands_both_knots_dry_fan;
#[path = "source_cases_three_bands_wide_bottom_cap_rest.rs"]
mod source_cases_planned_three_bands_uniform_trace_origin_before_native_face_190;
#[path = "source_cases_three_bands_wide_bottom_cap_rest_equivalents.rs"]
mod source_cases_planned_three_bands_uniform_trace_origin_before_retained_history_face_190;
#[path = "source_cases_three_bands_wide_bottom_uniform_velocity.rs"]
mod source_cases_planned_three_bands_uniform_uniform_cap_represented_tenth_forward;
#[path = "source_cases_three_bands_wide_bottom_rarefaction.rs"]
mod source_cases_planned_three_bands_wet_planned_three_bands_both_knots_near_pure_fan;
#[path = "source_cases_three_bands_wide_bottom_weak_cap_deficits.rs"]
mod source_cases_planned_three_bands_wet_trace_origin_after_retained_history_face_193;
#[path = "source_cases_three_bands_wide_bottom_weak_scalar_states.rs"]
mod source_cases_planned_three_bands_wet_trace_origin_before_native_face_197;
#[path = "source_cases_three_bands_wide_bottom_weak_expanded_states.rs"]
mod source_cases_planned_three_bands_wet_trace_origin_before_retained_history_face_194;
#[path = "source_cases_rectangle_dry_rectangle_right_dry_sonic.rs"]
mod source_cases_rectangle_dry_rectangle_right_dry_sonic;
#[path = "source_cases_rectangle_uniform_rectangle_uniform_rest.rs"]
mod source_cases_rectangle_uniform_rectangle_uniform_rest;
#[path = "source_cases_rectangle_wet_rectangle_depth_jump.rs"]
mod source_cases_rectangle_wet_rectangle_depth_jump;
#[path = "source_cases_rectangle_wet_rectangle_stationary_shock.rs"]
mod source_cases_rectangle_wet_rectangle_stationary_shock;
#[path = "source_cases_rectangle_wet_rectangle_vacuum.rs"]
mod source_cases_rectangle_wet_rectangle_vacuum;
#[path = "source_cases_rectangle_wet_retained_vacuum_inside_film.rs"]
mod source_cases_rectangle_wet_retained_vacuum_inside_film;
#[path = "source_cases_three_bands_dry_three_bands_both_knots_dry_fan.rs"]
mod source_cases_three_bands_dry_three_bands_both_knots_dry_fan;
#[path = "source_cases_three_bands_uniform_three_bands_uniform_upper.rs"]
mod source_cases_three_bands_uniform_three_bands_uniform_upper;
#[path = "source_cases_three_bands_tiny_upwind.rs"]
mod source_cases_three_bands_wet_captured_tiny_upwind_face_64;
#[path = "source_cases_three_bands_wet_three_bands_depth_jump.rs"]
mod source_cases_three_bands_wet_three_bands_depth_jump;
#[path = "source_cases_three_bands_wet_three_bands_loose_recovery_allowance.rs"]
mod source_cases_three_bands_wet_three_bands_loose_recovery_allowance;
#[path = "source_cases_trapezoid_uniform_trapezoid_uniform_flow.rs"]
mod source_cases_trapezoid_uniform_trapezoid_uniform_flow;
#[path = "source_cases_trapezoid_wet_trapezoid_depth_jump.rs"]
mod source_cases_trapezoid_wet_trapezoid_depth_jump;
#[path = "source_cases_triangle_dry_triangle_right_dry_sonic.rs"]
mod source_cases_triangle_dry_triangle_right_dry_sonic;
#[path = "source_cases_triangle_uniform_triangle_uniform_flow.rs"]
mod source_cases_triangle_uniform_triangle_uniform_flow;
#[path = "source_cases_triangle_wet_triangle_depth_jump.rs"]
mod source_cases_triangle_wet_triangle_depth_jump;
pub(super) use case_index::CASES;
pub(super) use model::{Case, Reference};
pub(super) use profiles::PROFILES;
