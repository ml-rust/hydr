//! Original immutable trajectory tensors and temporal support metadata.

mod context;
mod input;

pub(crate) use context::CpuTrajectoryContext;
pub(crate) use input::{CpuTrajectoryInput, TrajectoryPointMetadata};
