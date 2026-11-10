//! Tensor assembly from exclusive original point-source bindings.

mod context;
mod frames;
mod input;

pub(crate) use context::{
    CpuOperatingSourceContext, CpuOperatingSourceGroup, CpuOperatingSourceKind, CpuPrescribedSource,
};
pub(crate) use frames::frames;
pub(crate) use input::{CpuOperatingSourcePoint, CpuOperatingSourceValue, assemble};
