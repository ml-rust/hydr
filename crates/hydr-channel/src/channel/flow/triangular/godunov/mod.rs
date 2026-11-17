//! Exact barotropic Riemann sampling and certified flux accuracy.

mod accuracy;
mod analytic;
mod curves;
mod roots;
mod sample_bounds;
mod sampling;
mod types;

pub(in crate::channel::flow::triangular) use types::RiemannInput;
