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
#[allow(dead_code)]
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
#[path = "channel_pools/surveyed.rs"]
mod surveyed;
#[allow(dead_code)]
#[path = "channel_pools/surveyed_fixture.rs"]
mod surveyed_fixture;
#[path = "channel_pools/surveyed_inventory.rs"]
mod surveyed_inventory;
#[path = "channel_pools/surveyed_mode.rs"]
mod surveyed_mode;
#[allow(dead_code)]
#[path = "support/surveyed_profiles.rs"]
mod surveyed_profiles;
#[allow(dead_code)]
#[path = "channel_pools/surveyed_reference.rs"]
mod surveyed_reference;
#[path = "channel_pools/surveyed_rest.rs"]
mod surveyed_rest;
#[path = "channel_pools/surveyed_restoration.rs"]
mod surveyed_restoration;

#[test]
fn native_finite_pool_forcing_preserves_views_aliases_signed_zero_and_destination_restarts()
-> fixture::Result {
    native::execute::<runtime::Runtime>(
        runtime::client(),
        runtime::client(),
        native::Coverage::all(),
    )
}
