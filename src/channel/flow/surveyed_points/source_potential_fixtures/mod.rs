#[path = "depth_and_wave_transitions.rs"]
mod depth_and_wave_transitions;
#[path = "dry_and_vacuum_limits.rs"]
mod dry_and_vacuum_limits;
#[path = "dry_fans_across_sections.rs"]
mod dry_fans_across_sections;
#[path = "index.rs"]
mod index;
#[path = "knot_and_moving_capacity_states.rs"]
mod knot_and_moving_capacity_states;
#[path = "model.rs"]
mod model;
#[path = "near_vacuum_and_stationary_shocks.rs"]
mod near_vacuum_and_stationary_shocks;
#[path = "near_vacuum_and_upwind_sources.rs"]
mod near_vacuum_and_upwind_sources;
#[path = "rarefaction_and_small_capacity_changes.rs"]
mod rarefaction_and_small_capacity_changes;
#[path = "recovery_and_rarefaction_controls.rs"]
mod recovery_and_rarefaction_controls;
#[path = "retained_capacity_transitions.rs"]
mod retained_capacity_transitions;
#[path = "retained_capacity_velocities.rs"]
mod retained_capacity_velocities;
#[path = "retained_knots_and_capacity.rs"]
mod retained_knots_and_capacity;
#[path = "small_positive_sources.rs"]
mod small_positive_sources;
#[path = "surveyed_capacity_transitions.rs"]
mod surveyed_capacity_transitions;
#[path = "surveyed_capacity_velocities.rs"]
mod surveyed_capacity_velocities;
#[path = "surveyed_knots_and_capacity.rs"]
mod surveyed_knots_and_capacity;
#[path = "uniform_and_depth_transitions.rs"]
mod uniform_and_depth_transitions;
#[path = "uniform_basic_sections.rs"]
mod uniform_basic_sections;
#[path = "weak_transitions_and_rectangular_dry_states.rs"]
mod weak_transitions_and_rectangular_dry_states;

pub(in crate::channel::flow::surveyed_points) use index::CASES;
pub(in crate::channel::flow::surveyed_points) use model::{Case, Reference, SideReference};
