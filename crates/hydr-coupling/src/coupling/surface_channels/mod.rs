//! Couple the 2D surface engine with 1D channel reaches through lateral bank-overflow weirs.
//!
//! [`SurfaceChannelDefinition`] checks a [`SurfaceChannelSetup`] and synthesizes the
//! coordinator bindings. [`SurfaceChannelBatch`] builds the surface engine and every reach.
//! [`SurfaceChannelBatch::trial_to`] calculates one common interval as a
//! [`SurfaceChannelTrial`]. Accepting it replaces every child at once.
//!
//! Each interval iterates on end stages and signed link volumes from the accepted state:
//! - Midpoint heads give each weir's discharge over the interval.
//! - Surface donors offer their post-transport water. Channel donors offer their start water
//!   minus native face outflow and caller withdrawals.
//! - One factor per donor scales every link it feeds, so no donor gives more than it holds.
//! - The surface applies the summed volumes of each cell. Each reach applies each link's volume
//!   on its synthesized bindings.
//!
//! Surface water entering a reach carries the surface velocity along the reach tangent.
//! Channel water entering the surface carries the channel start velocity along the tangent.

mod accounting;
mod batch;
mod bindings;
mod context;
mod definition;
mod error;
mod evaluate;
mod exchange;
mod forcing;
mod iteration;
mod output;
mod projection;
mod start;
mod state;
mod trial;
mod validation;

pub use batch::SurfaceChannelBatch;
pub use definition::{
    SURFACE_CHANNEL_TANGENT_TOLERANCE, SurfaceChannelConfig, SurfaceChannelInitialState,
    SurfaceChannelLink, SurfaceChannelReach, SurfaceChannelSetup,
};
pub use error::SurfaceChannelError;
pub use forcing::{SurfaceChannelForcingInterval, SurfaceChannelReachForcing};
pub use output::{
    SurfaceChannelLinkLedger, SurfaceChannelLinkTransfer, SurfaceChannelStepDiagnostics,
    SurfaceChannelWaterBalance, SurfaceChannelWeirRegime,
};
pub use trial::SurfaceChannelTrial;
pub use validation::SurfaceChannelDefinition;
