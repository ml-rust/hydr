mod bank;
mod build;
pub(in crate::channel::section::prismatic) mod knot;
mod rows;
mod types;

pub(crate) use types::{BoundaryAreaTable, BoundaryAreas};

#[cfg(test)]
mod tests;
