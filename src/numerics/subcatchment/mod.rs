//! Conservative source stages, same-interval transfers, and complete tensor admission.

mod accounting;
mod admission;
mod phase;
mod soil;
mod sources;
mod step;
mod timing;
mod trial;

pub(crate) mod bounds;
pub(crate) mod bridge;
pub(crate) mod transfer;

pub(crate) use admission::check_step;
pub(crate) use step::{
    CpuSubareaStep, CpuSubcatchmentSoilStep, CpuSubcatchmentStep, CpuSubcatchmentTotals,
};
pub(crate) use trial::{advance, initial_storage_scale_lower};
