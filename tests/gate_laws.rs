//! Explicit gate operations, supported aperture geometry, and immutable proposals.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "gate_laws/admission.rs"]
mod admission;
#[path = "../examples/gate_laws_reference/comparison.rs"]
mod comparison;
#[path = "gate_laws/definitions.rs"]
mod definitions;
#[path = "gate_laws/fixture.rs"]
mod fixture;
#[path = "gate_laws/layouts.rs"]
mod layouts;
#[path = "gate_laws/lifecycle.rs"]
mod lifecycle;
#[path = "../examples/gate_laws_reference/reference_case.rs"]
mod reference_case;
#[path = "../examples/gate_laws_reference/inputs/mod.rs"]
mod reference_inputs;
#[path = "gate_laws/references.rs"]
mod references;
#[path = "gate_laws/support.rs"]
mod support;
#[path = "gate_laws/transitions.rs"]
mod transitions;
