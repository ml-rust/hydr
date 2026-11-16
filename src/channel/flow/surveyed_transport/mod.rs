//! Native surveyed reach transport, complete-wave bounds, and split resistance.

mod cfl;
mod diagnostics;
pub(super) mod error;
mod faces;
mod flux;
pub(super) mod friction;
pub(super) mod source_diagnostics;

#[cfg(test)]
mod friction_expansion_tests;
#[cfg(test)]
mod friction_test_data;
#[cfg(test)]
mod friction_tests;
