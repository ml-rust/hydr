//! Couple the 2D surface engine with 1D channel reaches through lateral bank-overflow weirs.
//!
//! [`SurfaceChannelDefinition`] checks a [`SurfaceChannelSetup`] and synthesizes the
//! coordinator bindings. [`SurfaceChannelBatch`] builds the surface engine and every reach.

mod batch;
mod bindings;
mod definition;
mod error;
mod validation;

pub use batch::SurfaceChannelBatch;
pub use definition::{
    SURFACE_CHANNEL_TANGENT_TOLERANCE, SurfaceChannelConfig, SurfaceChannelInitialState,
    SurfaceChannelLink, SurfaceChannelReach, SurfaceChannelSetup,
};
pub use error::SurfaceChannelError;
pub use validation::SurfaceChannelDefinition;
