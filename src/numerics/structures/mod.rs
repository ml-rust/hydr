//! Explicit typed structure-law evaluation on configured Numr CPU tensors.

mod admission;
mod direction;
mod evaluation;
pub(crate) mod held;
mod orifice;
mod weir;

pub(crate) use evaluation::evaluate;
