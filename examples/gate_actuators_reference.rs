//! Original-bit actuator histories with independently checked physical receipts.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

#[path = "support/gate_actuators/mod.rs"]
mod reference;

fn main() -> reference::Result {
    reference::run()
}
