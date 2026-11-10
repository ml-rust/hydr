//! Physical acceptance cases for rectangular one-dimensional channel flow.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[allow(dead_code)]
#[path = "support/analytic.rs"]
mod analytic;

#[path = "channel_flow/admission.rs"]
mod admission;
#[path = "channel_flow/common.rs"]
mod common;
#[path = "channel_flow/refinement.rs"]
mod refinement;
#[path = "channel_flow/transport.rs"]
mod transport;
