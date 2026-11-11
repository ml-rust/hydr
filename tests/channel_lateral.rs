//! Physical source and continuation checks for rectangular channel reaches.

#[path = "channel_lateral/admission.rs"]
mod admission;
#[path = "channel_lateral/arithmetic.rs"]
mod arithmetic;
#[allow(dead_code)]
#[path = "channel_flow/common.rs"]
mod common;
#[path = "channel_lateral/fixtures.rs"]
mod fixtures;
#[path = "channel_lateral/intervals.rs"]
mod intervals;
#[path = "channel_lateral/manufactured.rs"]
mod manufactured;
#[path = "channel_lateral/physical.rs"]
mod physical;
#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;
