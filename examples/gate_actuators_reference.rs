//! Original-bit actuator histories with independently checked physical receipts.

#[path = "support/gate_actuators/mod.rs"]
mod reference;

fn main() -> reference::Result {
    reference::run()
}
