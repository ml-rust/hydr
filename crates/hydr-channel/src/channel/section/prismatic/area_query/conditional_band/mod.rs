#[cfg(test)]
mod clipped_endpoint_tests;
mod clipped_endpoints;
mod clipping;
mod endpoint_authority;
mod evaluation;
mod integration_geometry;
mod preparation;
mod range;
mod selection;
mod target_coordinates;
mod types;
mod wet_depth_geometry;

pub(crate) use clipped_endpoints::ClippedBandEndpoints;
pub(crate) use types::{
    ConditionalBandPanel, ConditionalBandSegments, PrefixIntegrationGeometry, PreparedAreaPrefix,
};
pub(crate) use wet_depth_geometry::WetDepthGeometry;

#[cfg(test)]
mod oracle_tests;

#[cfg(test)]
mod prefix_tests;

#[cfg(test)]
mod tiny_gap_tests;

#[cfg(test)]
mod wet_depth_geometry_tests;

#[cfg(test)]
mod wet_depth_potential_tests;
