//! Frozen sampled-control and gate-response physical reference observations.

#[path = "controlled_gates_reference/mod.rs"]
mod reference;

fn main() -> reference::Result {
    reference::run()
}
