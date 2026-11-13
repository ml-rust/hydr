pub(super) mod fields;
pub(super) mod inverse;

#[cfg(test)]
mod admission;
#[cfg(test)]
mod area_expression;
#[cfg(test)]
mod boundaries;
#[cfg(test)]
mod chart;
#[cfg(test)]
pub(crate) mod conditional_band;
#[cfg(test)]
mod endpoints;
#[cfg(test)]
pub(super) mod evaluation;
#[cfg(test)]
mod pressure;
#[cfg(test)]
pub(crate) use pressure::{PressureEndpoint, pressure_endpoint};
#[cfg(test)]
mod range;
#[cfg(test)]
mod selection;
#[cfg(test)]
pub(super) use admission::metadata as selection_metadata;
#[cfg(test)]
pub(super) use selection::{
    coordinates as select_coordinates, expansion as select_expansion, interval as select_interval,
};
#[cfg(test)]
mod source_difference;
#[cfg(test)]
mod support;
#[cfg(test)]
mod support_assessment;
#[cfg(test)]
mod support_selection;
#[cfg(test)]
pub(crate) use support_selection::SupportSelection;

#[cfg(test)]
mod support_selection_tests;

#[cfg(test)]
pub(crate) use chart::{AreaAnchor, AreaDirection, AreaQuery, SupportedAreaQuery};
#[cfg(test)]
pub(crate) use endpoints::PhysicalEndpoint;
#[cfg(test)]
pub(crate) use evaluation::AreaQueryGeometry;
#[cfg(test)]
pub(crate) use source_difference::source_difference;

#[cfg(test)]
pub(super) mod tests;

#[cfg(test)]
mod source_difference_tests;

#[cfg(test)]
mod endpoint_tests;

#[cfg(test)]
mod provenance_tests;
