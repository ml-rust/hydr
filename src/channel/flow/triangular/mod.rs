//! Prismatic triangular conserved face arithmetic.

pub(in crate::channel::flow) mod envelope;
mod faces;
mod flux;
mod godunov;
mod pressure;
pub(in crate::channel::flow::triangular) mod wave;

#[cfg(test)]
mod tests;
