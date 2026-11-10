//! Tensor assembly from exclusive original point-source bindings.

mod context;
mod frames;
mod input;

pub(crate) use context::{
    TensorOperatingSourceContext, TensorOperatingSourceGroup, TensorOperatingSourceKind,
    TensorPrescribedSource,
};
pub(crate) use frames::frames;
pub(crate) use input::{TensorOperatingSourcePoint, TensorOperatingSourceValue, assemble};
