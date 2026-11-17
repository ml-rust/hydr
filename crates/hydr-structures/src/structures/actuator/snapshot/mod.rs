//! Complete trusted and caller-owned actuator restoration.

mod compatibility;
mod package;
mod proof_check;
mod receipt_bits;
mod restore;
mod state_check;

pub use package::GateActuatorSnapshot;
