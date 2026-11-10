//! Actuator reference input, event progression, and receipt projection.

mod context;
mod history;
mod inputs;
mod observer;
mod proof;
mod row;
mod state;

pub use context::Result;
pub use observer::run;
