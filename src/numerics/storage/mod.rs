//! Closed storage arithmetic and admission over configured persistent CPU tensors.

pub(crate) mod accounting;
mod admission;
pub(crate) mod bounds;
pub(crate) mod curves;
mod trial;

pub(crate) use admission::check_step;
pub(crate) use trial::advance;
