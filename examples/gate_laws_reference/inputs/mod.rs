//! Original-F64 aperture inputs and independent discharge references.

mod catalog;
mod extremes;
mod geometry;
mod ordinary;
mod transitions;
mod types;

pub use catalog::cases;
pub use extremes::EXTREMES;
pub use geometry::GEOMETRY;
pub use ordinary::ORDINARY;
pub use transitions::TRANSITIONS;
pub use types::{GateReferenceCase, ReferenceExpectation, ReferenceHead};
