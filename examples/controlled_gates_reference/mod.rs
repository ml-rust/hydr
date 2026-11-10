//! Original inputs, atomic event progression, and explicit child receipt projections.

#[path = "../support/gate_actuators/state.rs"]
mod actuator_state;
mod context;
mod control;
mod history;
mod inputs;
mod observer;
#[path = "../support/gate_actuators/proof.rs"]
mod proof;
mod row;

pub use context::Result;
pub use observer::run;
