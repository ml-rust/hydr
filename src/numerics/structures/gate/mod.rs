//! Hydrostatic rectangular-aperture proposals with original-opening geometry.

mod classification;
mod evaluation;
mod geometry;
mod geometry_bounds;

pub(crate) use evaluation::evaluate;

pub(crate) mod trajectory;
