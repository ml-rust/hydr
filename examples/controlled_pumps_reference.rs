//! Original-bit sampled pump response through registered dynamic operating sources.

#[path = "controlled_pumps_reference/mod.rs"]
mod reference;

fn main() -> reference::Result {
    reference::run()
}
