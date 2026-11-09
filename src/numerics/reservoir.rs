//! Conservative implicit reservoir integration and reusable receipt admission.

mod advance;
mod algebra;
mod check;
mod receipt;

pub(crate) use advance::advance;
pub(crate) use algebra::EXPONENT;
pub(crate) use check::check_step;
pub(crate) use receipt::CpuReservoirStep;

#[cfg(test)]
mod tests;
