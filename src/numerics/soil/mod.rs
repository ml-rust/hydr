//! Native Modified Green–Ampt state transitions and hydraulic integration.

mod integration;
pub(crate) mod roundoff;
pub(crate) mod transition;

pub(crate) use transition::advance;
