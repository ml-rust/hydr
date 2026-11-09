//! Native Modified Green–Ampt state transitions and hydraulic integration.

mod integration;
mod transition;

pub(crate) use transition::advance;
