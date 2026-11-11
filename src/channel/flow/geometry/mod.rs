mod definition;
mod tensor;

pub use definition::{ChannelGeometry, TriangularChannelCell};
pub(in crate::channel::flow) use tensor::{TensorGeometry, TriangleGeometry};

#[cfg(test)]
mod definition_tests;
