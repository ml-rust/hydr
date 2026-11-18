//! Prismatic triangular conserved face arithmetic.

mod curves;
pub(in crate::channel::flow) mod envelope;
mod faces;
#[cfg(test)]
mod fixture;
mod flux;
mod fronts;
mod godunov;
mod pressure;
pub(in crate::channel::flow::triangular) mod wave;
