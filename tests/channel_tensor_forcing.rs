//! Native channel forcing transport and atomic admission acceptance.

#[path = "channel_tensor_forcing/admission.rs"]
mod admission;
#[allow(dead_code)]
#[path = "channel_boundaries/common.rs"]
mod boundary_common;
#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;
#[path = "channel_tensor_forcing/snapshot_bits.rs"]
mod snapshot_bits;
#[path = "channel_tensor_forcing/transport.rs"]
mod transport;
