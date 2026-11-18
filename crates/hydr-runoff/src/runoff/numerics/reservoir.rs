//! Conservative implicit reservoir integration and reusable receipt admission.

mod advance;
mod algebra;
mod check;
pub(crate) mod drainage;
#[cfg(test)]
mod fixture;
mod receipt;

pub(crate) use advance::advance;
pub(crate) use check::check_step;
pub(crate) use drainage::EXPONENT;
pub(crate) use receipt::TensorReservoirStep;
