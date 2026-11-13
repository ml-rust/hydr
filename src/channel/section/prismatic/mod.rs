//! Certified surveyed geometry on represented depths and conserved areas.

mod area_point_fields;
mod area_point_input;
mod area_point_inverse;
mod area_point_projection;
mod area_point_query;
mod area_point_recovery;
#[cfg(test)]
mod area_point_selection;
#[cfg(test)]
mod area_point_selection_tests;
mod area_query;
mod band_geometry;
mod batch;
#[cfg(test)]
mod boundary_area;
mod compilation;
mod compilation_bounds;
mod config;
mod context;
mod evaluation;
mod inputs;
mod inverse_bands;
mod output;
mod perimeter;
mod projection;
mod query;
mod query_input;
mod recovery;
mod topology;

pub use area_point_query::{
    PrismaticAreaPointBandCoordinates, PrismaticAreaPointBandProperties,
    PrismaticAreaPointRecoveredProperties, PrismaticAreaPointRecovery,
};
pub use batch::PrismaticSectionBatch;
pub use config::{PRISMATIC_BAND_WORK_LIMIT, PrismaticRecoveryConfig};
pub use output::{PrismaticRecoveredProperties, PrismaticSectionProperties};
pub use perimeter::{
    PrismaticPerimeterAreaPointRecoveredProperties, PrismaticPerimeterAreaPointRecovery,
    PrismaticPerimeterBatch, PrismaticPerimeterProperties, PrismaticPerimeterQuery,
    PrismaticPerimeterRecoveredProperties, PrismaticPerimeterRecovery,
};
pub use query::{PrismaticSectionQuery, PrismaticSectionRecovery};

mod hydraulic_evaluation;
mod hydraulic_geometry;
mod riemann;
pub(crate) use hydraulic_geometry::{Constitutive, GatheredBands, HydraulicGeometry};

#[cfg(test)]
mod hydraulic_geometry_tests;
