//! Original immutable trajectory tensors and temporal support metadata.

mod context;
mod input;

pub(crate) use context::{TensorTrajectoryContext, indices};
pub(crate) use input::{TensorTrajectoryInput, TrajectoryPointMetadata};
