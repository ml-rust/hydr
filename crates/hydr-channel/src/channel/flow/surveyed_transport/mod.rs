//! Native surveyed reach transport, direct end faces, complete-wave bounds, and split resistance.

mod cfl;
mod diagnostics;
mod direct;
pub(super) mod error;
mod faces;
mod flux;
pub(super) mod friction;
mod port;
pub(super) mod source_diagnostics;

#[cfg(test)]
mod friction_test_data;
