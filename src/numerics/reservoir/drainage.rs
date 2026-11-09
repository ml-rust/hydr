//! Reusable hydraulic budgets, excess roots, and represented conservative export.

mod admission;
mod algebra;
mod budget;
mod receipt;
mod solve;

pub(crate) use admission::admit_drainage;
pub(in crate::numerics::reservoir) use admission::{check_residual, equal};
pub(crate) use algebra::EXPONENT;
pub(in crate::numerics::reservoir) use algebra::{check_metadata, volume};
pub(crate) use budget::{DrainageBudget, prepare_budget};
pub(crate) use receipt::{DrainageResult, DrainageRoot};
pub(crate) use solve::drain;
