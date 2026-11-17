pub(super) mod fields;
pub(super) mod inverse;

mod admission;
mod area_expression;
mod boundaries;
mod chart;
mod domain_coverage;
pub(crate) use domain_coverage::cover_domain;
#[cfg(test)]
pub(crate) use domain_coverage::{CoveredDomain, DomainCoverage};
pub(crate) mod conditional_band;
#[cfg(test)]
mod domain_coverage_tests;
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
#[cfg(test)]
pub(crate) use support_selection::SupportSelection;

#[cfg(test)]
mod support_selection_tests;

pub(crate) use chart::{AreaAnchor, AreaDirection, AreaQuery, SupportedAreaQuery};
pub(crate) use endpoints::PhysicalEndpoint;
pub(crate) use evaluation::AreaQueryGeometry;
pub(crate) use source_difference::source_difference;

#[cfg(test)]
pub(super) mod tests;

#[cfg(test)]
mod source_difference_tests;

#[cfg(test)]
mod endpoint_tests;

#[cfg(test)]
mod provenance_tests;
