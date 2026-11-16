//! Piecewise-linear cross-section properties without flow advancement.

mod admission;
mod batch;
mod context;
mod definition;
mod error;
mod evaluation;
mod output;
mod prismatic;
mod query;
mod recovery;
mod recovery_config;
mod recovery_query;
mod segments;

#[cfg(test)]
mod native_tests;
#[cfg(test)]
mod recovery_native_tests;

pub use batch::CrossSectionBatch;
pub use definition::{CrossSectionDefinition, CrossSectionPoint};
pub use error::CrossSectionError;
pub use output::CrossSectionProperties;
pub use prismatic::{
    PRISMATIC_BAND_WORK_LIMIT, PrismaticAreaPointBandCoordinates, PrismaticAreaPointBandProperties,
    PrismaticAreaPointRecoveredProperties, PrismaticAreaPointRecovery,
    PrismaticPerimeterAreaPointRecoveredProperties, PrismaticPerimeterAreaPointRecovery,
    PrismaticPerimeterBatch, PrismaticPerimeterProperties, PrismaticPerimeterQuery,
    PrismaticPerimeterRecoveredProperties, PrismaticPerimeterRecovery,
    PrismaticRecoveredProperties, PrismaticRecoveryConfig, PrismaticSectionBatch,
    PrismaticSectionProperties, PrismaticSectionQuery, PrismaticSectionRecovery,
};
pub use query::CrossSectionQuery;
pub use recovery_config::CrossSectionRecoveryConfig;
pub use recovery_query::{CrossSectionRecoveredProperties, CrossSectionRecovery};

pub(crate) use prismatic::PressureEndpoint;
pub(crate) use prismatic::{Constitutive, GatheredBands, HydraulicGeometry};

pub(crate) use prismatic::{
    AreaAnchor, AreaDirection, AreaQuery, BoundaryAreas, ClippedBandEndpoints,
    ConditionalBandSegments, PhysicalEndpoint, PreparedAreaPrefix, SupportedAreaQuery,
    WetDepthGeometry, cover_domain, pressure_endpoint, source_difference,
};
