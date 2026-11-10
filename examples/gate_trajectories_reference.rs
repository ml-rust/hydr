//! Original-bit trajectory admissions, receipts, and explicit operation projections.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

#[path = "gate_trajectories_reference/observer.rs"]
mod observer;
#[path = "gate_trajectories_reference/receipt.rs"]
mod receipt;
#[path = "gate_trajectories_reference/reference_case.rs"]
mod reference_case;
#[path = "gate_trajectories_reference/inputs/mod.rs"]
mod reference_inputs;

fn main() -> reference_case::Result {
    observer::run()
}
