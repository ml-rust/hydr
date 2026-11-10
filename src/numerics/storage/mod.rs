//! Closed storage arithmetic and admission over configured persistent runtime tensors.

pub(crate) mod accounting;
mod admission;
pub(crate) mod bounds;
pub(crate) mod curves;
mod source_accounting;
mod source_area;
mod source_caps;
pub(crate) mod sources;
mod trial;

pub(crate) use admission::check_step;
pub(crate) use trial::advance;
pub(crate) use trial::advance_with_sources;
