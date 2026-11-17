//! Reusable hydraulic budgets, excess roots, and represented conservative export.

mod admission;
mod algebra;
mod budget;
mod receipt;
mod solve;

pub(crate) use admission::admit_drainage;
pub(crate) use admission::check_residual;
pub(in crate::runoff::numerics::reservoir) use admission::equal;
pub(crate) use algebra::EXPONENT;
pub(crate) use algebra::{check_metadata, volume};
pub(crate) use budget::{DrainageBudget, prepare_budget};
pub(crate) use receipt::{DrainageResult, DrainageRoot};
pub(crate) use solve::drain;
