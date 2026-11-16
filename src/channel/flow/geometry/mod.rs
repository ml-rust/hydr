mod definition;
mod surveyed;
mod surveyed_config;
mod tensor;

pub use definition::{ChannelGeometry, SurveyedChannelCell, TriangularChannelCell};
pub(in crate::channel::flow) use surveyed::SurveyedGeometry;
pub use surveyed_config::ChannelSurveyedConfig;
pub(in crate::channel::flow) use tensor::{TensorGeometry, TriangleGeometry};

#[cfg(test)]
mod definition_tests;
