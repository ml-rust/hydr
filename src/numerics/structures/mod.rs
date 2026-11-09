//! Explicit typed structure-law evaluation on configured Numr CPU tensors.

mod admission;
mod bounds;
mod direction;
mod evaluation;
pub(crate) mod gate;
pub(crate) mod held;
mod orifice;
pub(crate) mod pump;
mod weir;

pub(crate) use evaluation::evaluate;
