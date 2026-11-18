pub(super) mod fields;
pub(super) mod inverse;

mod admission;
mod area_expression;
mod boundaries;
mod chart;
mod domain_coverage;
pub(crate) use domain_coverage::cover_domain;
pub(crate) mod conditional_band;
mod endpoints;
pub(super) mod evaluation;
pub(super) mod hydrostatic_point;
mod pressure;
pub(crate) use pressure::{PressureEndpoint, pressure_endpoint};
mod range;
mod selection;
pub(super) use admission::metadata as selection_metadata;
pub(super) use selection::{
    coordinates as select_coordinates, expansion as select_expansion, interval as select_interval,
};
mod source_difference;
mod support;
mod support_assessment;
mod support_selection;

pub(crate) use chart::{AreaAnchor, AreaDirection, AreaQuery, SupportedAreaQuery};
pub(crate) use endpoints::PhysicalEndpoint;
pub(crate) use evaluation::AreaQueryGeometry;
pub(crate) use source_difference::source_difference;

#[cfg(test)]
pub(in crate::channel::section::prismatic) mod fixtures;

#[cfg(test)]
mod query_fixture;
mod rebinding;

#[cfg(test)]
pub(in crate::channel::section::prismatic) mod test_support;
