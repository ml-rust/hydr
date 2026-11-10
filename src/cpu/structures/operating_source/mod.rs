//! Tensor assembly from exclusive original point-source bindings.

mod context;
mod input;

pub(crate) use context::{
    CpuOperatingSourceContext, CpuOperatingSourceGroup, CpuOperatingSourceKind, CpuPrescribedSource,
};
pub(crate) use input::{CpuOperatingSourcePoint, CpuOperatingSourceValue, assemble, frames};
