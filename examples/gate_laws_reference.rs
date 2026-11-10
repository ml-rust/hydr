//! Original-F64 rectangular-aperture discharge and geometry observer.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

#[path = "gate_laws_reference/comparison.rs"]
mod comparison;
#[path = "gate_laws_reference/observer.rs"]
mod observer;
#[path = "gate_laws_reference/reference_case.rs"]
mod reference_case;
#[path = "gate_laws_reference/inputs/mod.rs"]
mod reference_inputs;

fn main() -> reference_case::Result {
    observer::run()
}
