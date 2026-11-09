//! Oriented pump reference/affinity queries with complete normalization uncertainty admission.

mod admission;
mod bounds;
mod curve;
mod evaluation;
mod normalization;

pub(crate) use evaluation::evaluate;
