//! Surveyed reach accuracy, sources, admission, and complete continuation.

#[path = "support/runtime.rs"]
mod runtime;

#[path = "channel_surveyed/admission.rs"]
mod admission;
#[path = "channel_surveyed/endpoints_admission.rs"]
mod endpoints_admission;
#[path = "channel_surveyed/endpoints_identity.rs"]
mod endpoints_identity;
#[path = "channel_surveyed/endpoints_propagation.rs"]
mod endpoints_propagation;
#[allow(dead_code)]
#[path = "channel_surveyed/endpoints_propagation_reference.rs"]
mod endpoints_propagation_reference;
#[allow(dead_code)]
#[path = "channel_surveyed/endpoints_reference.rs"]
mod endpoints_reference;
#[path = "channel_surveyed/endpoints_rest.rs"]
mod endpoints_rest;
#[path = "channel_surveyed/endpoints_restart.rs"]
mod endpoints_restart;
#[path = "channel_surveyed/endpoints_support.rs"]
mod endpoints_support;
#[path = "channel_surveyed/endpoints_traces.rs"]
mod endpoints_traces;
#[path = "channel_surveyed/fixtures.rs"]
mod fixtures;
#[path = "channel_surveyed/lifecycle.rs"]
mod lifecycle;
#[path = "channel_surveyed/native.rs"]
mod native;
#[path = "channel_surveyed/physics.rs"]
mod physics;
#[path = "channel_surveyed/sources.rs"]
mod sources;
#[path = "channel_surveyed/support.rs"]
mod support;
#[allow(dead_code)]
#[path = "support/surveyed_profiles.rs"]
mod surveyed_profiles;
