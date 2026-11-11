//! Physical and lifecycle acceptance for actual prismatic triangular reaches.

#[path = "channel_triangular/admission.rs"]
mod admission;
#[path = "channel_triangular/common.rs"]
mod common;
#[path = "channel_triangular/lifecycle.rs"]
mod lifecycle;
#[path = "channel_triangular/refinement.rs"]
mod refinement;
#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;
#[path = "channel_triangular/transport.rs"]
mod transport;
#[path = "support/triangle_reference.rs"]
mod triangle_reference;
