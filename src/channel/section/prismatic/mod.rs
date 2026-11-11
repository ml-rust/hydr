//! Certified surveyed geometry on represented depths and conserved areas.

mod band_geometry;
mod batch;
mod compilation;
mod compilation_bounds;
mod config;
mod context;
mod evaluation;
mod inputs;
mod inverse_bands;
mod output;
mod projection;
mod query;
mod query_input;
mod recovery;
mod topology;

pub use batch::PrismaticSectionBatch;
pub use config::{PRISMATIC_BAND_WORK_LIMIT, PrismaticRecoveryConfig};
pub use output::{PrismaticRecoveredProperties, PrismaticSectionProperties};
pub use query::{PrismaticSectionQuery, PrismaticSectionRecovery};
