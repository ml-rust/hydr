//! Finite channel/pool physics, topology, and atomic continuation acceptance.

#[path = "channel_pools/admission.rs"]
mod admission;
#[path = "channel_pools/fixture.rs"]
mod fixture;
#[path = "channel_pools/identity.rs"]
mod identity;
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

#[test]
fn native_finite_pool_forcing_preserves_views_aliases_signed_zero_and_destination_restarts()
-> fixture::Result {
    native::execute::<runtime::Runtime>(runtime::client(), runtime::client())
}
