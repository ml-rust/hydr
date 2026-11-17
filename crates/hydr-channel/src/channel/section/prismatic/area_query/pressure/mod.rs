mod bottom;
mod range;
mod secant;
mod types;
pub(crate) use types::PressureEndpoint;
mod weights;

pub(crate) use range::pressure_endpoint;
#[cfg(test)]
pub(crate) use range::pressure_range;

#[cfg(test)]
pub(super) mod tests;
