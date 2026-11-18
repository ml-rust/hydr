mod bottom;
mod range;
mod secant;
mod types;
pub(crate) use types::PressureEndpoint;
mod weights;

pub(crate) use range::pressure_endpoint;

#[cfg(test)]
#[path = "../pressure_fixtures/mod.rs"]
pub(in crate::channel::section::prismatic::area_query) mod fixtures;
