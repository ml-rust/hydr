use super::super::{source_tests::support as frozen, wet_depth::tests::support};
use frozen::hydraulic;

#[path = "point_flux_fixture_coherence.rs"]
pub(in crate::channel::flow::surveyed_points) mod coherence;
#[path = "point_flux_fixture_diagnostics.rs"]
pub(in crate::channel::flow::surveyed_points) mod diagnostics;
