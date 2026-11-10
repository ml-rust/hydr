//! Original-bit sampled-control histories and exact timer witness observations.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

#[path = "operating_controls_reference/check.rs"]
mod check;
#[path = "operating_controls_reference/history.rs"]
mod history;
#[path = "operating_controls_reference/observer.rs"]
mod observer;
#[path = "operating_controls_reference/reference_case.rs"]
mod reference_case;
#[path = "operating_controls_reference/inputs/mod.rs"]
mod reference_inputs;
#[path = "operating_controls_reference/row.rs"]
mod row;

fn main() -> reference_case::Result {
    observer::run()
}
