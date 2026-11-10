//! Persistent positive-area groups and complete accepted component state.

mod context;
mod group;
mod input;
mod state;

pub(crate) use context::TensorSubcatchmentContext;
pub(crate) use group::TensorSubareaGroup;
pub(crate) use input::TensorSubcatchmentInput;
pub(crate) use state::TensorSubcatchmentState;
