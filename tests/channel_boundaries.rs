//! Frozen physical and lifecycle acceptance for scalar and flux port channel endpoints.

#[path = "channel_boundaries/admission.rs"]
mod admission;
#[path = "channel_boundaries/common.rs"]
mod common;
#[path = "channel_boundaries/flux_port.rs"]
mod flux_port;
#[path = "channel_boundaries/flux_port_lifecycle.rs"]
mod flux_port_lifecycle;
#[path = "channel_boundaries/flux_port_surveyed.rs"]
mod flux_port_surveyed;
#[path = "channel_boundaries/lifecycle.rs"]
mod lifecycle;
#[path = "channel_boundaries/packing.rs"]
mod packing;
#[path = "channel_boundaries/propagation.rs"]
mod propagation;
#[allow(dead_code)]
#[path = "support/channel_boundary_reference.rs"]
mod reference;
#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;
#[allow(dead_code)]
#[path = "support/surveyed_profiles.rs"]
mod surveyed_profiles;
#[path = "channel_boundaries/traces.rs"]
mod traces;
