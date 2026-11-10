//! Configured persistent typed-structure coefficient and proposal tensors.

pub(crate) mod codes;
mod context;
pub(crate) mod control;
pub(crate) mod gate;
mod input;
mod operations;
pub(crate) mod power_metadata;
mod proposal;
pub(crate) mod pump;

pub(crate) use context::CpuStructureContext;
pub(crate) use input::CpuStructureInput;
pub(crate) use proposal::CpuStructureResult;
