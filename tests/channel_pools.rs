//! Finite channel/pool physics, topology, and atomic continuation acceptance.

#[path = "channel_pools/admission.rs"]
mod admission;
#[path = "channel_pools/fixture.rs"]
mod fixture;
#[path = "channel_pools/lifecycle.rs"]
mod lifecycle;
#[path = "channel_pools/linear_mode.rs"]
mod linear_mode;
#[path = "channel_pools/native.rs"]
mod native;
#[path = "channel_pools/physics.rs"]
mod physics;
#[path = "channel_pools/raised.rs"]
mod raised;
#[path = "channel_pools/reference.rs"]
mod reference;
#[path = "channel_pools/restoration.rs"]
mod restoration;
#[path = "support/runtime.rs"]
mod runtime;
