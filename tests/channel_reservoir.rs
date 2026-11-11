//! Frozen reservoir endpoint physics and continuation acceptance.

#[path = "channel_reservoir/admission.rs"]
mod admission;
#[allow(dead_code)]
#[path = "channel_boundaries/common.rs"]
mod boundary_common;
#[path = "channel_reservoir/fixtures.rs"]
mod fixtures;
#[path = "channel_reservoir/native.rs"]
mod native;
#[path = "channel_reservoir/physics.rs"]
mod physics;
#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;
#[path = "channel_tensor_forcing/snapshot_bits.rs"]
mod snapshot_bits;
#[path = "channel_tensor_forcing/transport.rs"]
mod transport;
