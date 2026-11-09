//! Configured persistent typed-structure coefficient and proposal tensors.

pub(crate) mod codes;
mod context;
mod input;
mod proposal;

pub(crate) use context::CpuStructureContext;
pub(crate) use input::CpuStructureInput;
pub(crate) use proposal::CpuStructureResult;
