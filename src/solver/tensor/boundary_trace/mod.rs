//! Wet-subcritical physical traces with prescribed stage or outward discharge.

mod admission;
mod discharge;
mod trace;
mod transmissive;
mod types;

pub(crate) use trace::characteristic_trace;
pub(crate) use transmissive::check_transmissive;
pub(crate) use types::{TensorBoundaryRequest, TensorBoundaryTrace};

#[cfg(test)]
mod tests;
