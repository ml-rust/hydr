//! Piecewise-linear cross-section properties without flow advancement.

mod admission;
mod batch;
mod context;
mod definition;
mod error;
mod evaluation;
mod output;
mod query;
mod segments;

#[cfg(test)]
mod native_tests;

pub use batch::CrossSectionBatch;
pub use definition::{CrossSectionDefinition, CrossSectionPoint};
pub use error::CrossSectionError;
pub use output::CrossSectionProperties;
pub use query::CrossSectionQuery;
