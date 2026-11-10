//! Frozen sampled-control and gate-response physical reference observations.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

#[path = "controlled_gates_reference/mod.rs"]
mod reference;

fn main() -> reference::Result {
    reference::run()
}
