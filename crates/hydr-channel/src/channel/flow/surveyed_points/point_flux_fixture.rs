use super::super::{source_fixture_support as frozen, wet_depth::fixture as support};
use frozen::hydraulic;

#[path = "point_flux_fixture_coherence.rs"]
pub(in crate::channel::flow::surveyed_points) mod coherence;
#[path = "point_flux_fixture_diagnostics.rs"]
pub(in crate::channel::flow::surveyed_points) mod diagnostics;
