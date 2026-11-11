//! Frozen original-bit geometry operands and exact reference enclosures.

mod forward;
mod forward_cap_support;
mod forward_conditioning;
mod forward_shapes;
mod forward_translated;
mod inverse;
mod inverse_cap_support;
mod inverse_explicit_areas;
mod inverse_multiband;
mod inverse_polygonal;
mod inverse_translated;
mod model;
mod profiles;
mod rejections;

pub(crate) use forward::FORWARD_CASES;
pub(crate) use inverse::INVERSE_CASES;
pub(crate) use model::*;
pub(crate) use profiles::{PROFILES, QUERY_BAND_LIMIT};
pub(crate) use rejections::{ADMISSION_REJECTIONS, QUERY_REJECTIONS, UNRESOLVED_INVERSE};
