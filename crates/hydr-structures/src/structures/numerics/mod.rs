//! Explicit typed structure-law evaluation on configured Numr runtime tensors.

mod admission;
mod bounds;
mod direction;
mod evaluation;
pub(crate) mod gate;
pub(crate) mod held;
mod log_add_exp;
mod orifice;
pub(crate) mod pump;
mod weir;

pub(crate) use evaluation::evaluate;

pub(crate) mod control;

pub(crate) mod actuator;

mod proposal;
pub(crate) use proposal::evaluate as proposal;
