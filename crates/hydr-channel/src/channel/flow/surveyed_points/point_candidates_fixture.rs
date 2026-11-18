use super::super::{source_fixture_support as frozen, wet_depth::fixture as support};
use frozen::{Result, hydraulic};

#[path = "point_candidates_fixture_diagnostics.rs"]
pub(in crate::channel::flow::surveyed_points) mod diagnostics;
#[path = "point_candidates_fixture_model.rs"]
pub(in crate::channel::flow::surveyed_points) mod fixture_model;
#[path = "point_candidates_fixtures.rs"]
pub(in crate::channel::flow::surveyed_points) mod fixtures;
#[path = "point_flux_fixture_model.rs"]
pub(in crate::channel::flow::surveyed_points) mod flux_fixture_model;
#[path = "point_flux_fixtures.rs"]
pub(in crate::channel::flow::surveyed_points) mod flux_fixtures;
#[path = "point_candidates_fixture_observed.rs"]
pub(in crate::channel::flow::surveyed_points) mod observed;
#[path = "point_candidates_fixture_wave_references.rs"]
pub(in crate::channel::flow::surveyed_points) mod wave_references;
