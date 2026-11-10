//! Original-bit sampled pump response through registered dynamic operating sources.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

#[path = "controlled_pumps_reference/mod.rs"]
mod reference;

fn main() -> reference::Result {
    reference::run()
}
