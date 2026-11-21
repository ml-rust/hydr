mod clipped_endpoints;
mod clipping;
mod endpoint_authority;
mod evaluation;
mod integration_geometry;
#[cfg(test)]
pub(crate) mod node_fixture;
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
