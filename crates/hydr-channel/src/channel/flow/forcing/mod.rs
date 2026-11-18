//! Original endpoint and lateral operands without a captured runtime client.

mod admission;
mod binding;
mod endpoint;
mod native;
mod ownership;
mod port;
mod projection;

pub(super) use native::{BoundChannelForcing, BoundLateralInputs};

pub use native::{
    ChannelTensorBoundary, ChannelTensorForcingInterval, ChannelTensorLateralInput,
    ChannelTensorWaterAmount,
};
pub use port::ChannelTensorFluxPort;
