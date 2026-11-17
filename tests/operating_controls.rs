//! Sampled command histories, exact timers, and complete transactional restoration.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "operating_controls/admission.rs"]
mod admission;
#[path = "../examples/operating_controls_reference/check.rs"]
mod check;
#[path = "operating_controls/fixture.rs"]
mod fixture;
#[path = "operating_controls/histories.rs"]
mod histories;
#[path = "operating_controls/identity.rs"]
mod identity;
#[path = "operating_controls/layouts.rs"]
mod layouts;
#[path = "operating_controls/lifecycle.rs"]
mod lifecycle;
#[path = "../examples/operating_controls_reference/reference_case.rs"]
mod reference_case;
#[path = "../examples/operating_controls_reference/inputs/mod.rs"]
mod reference_inputs;
#[path = "operating_controls/restoration.rs"]
mod restoration;
#[path = "operating_controls/snapshots.rs"]
mod snapshots;
