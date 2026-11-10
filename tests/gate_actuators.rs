//! Ideal finite-rate response, causal targets, and complete transactional restoration.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "gate_actuators/admission.rs"]
mod admission;
#[path = "gate_actuators/fixture.rs"]
mod fixture;
#[path = "gate_actuators/historical_completions.rs"]
mod historical_completions;
#[path = "gate_actuators/histories.rs"]
mod histories;
#[path = "../examples/support/gate_actuators/inputs.rs"]
mod inputs;
#[path = "gate_actuators/layouts.rs"]
mod layouts;
#[path = "gate_actuators/lifecycle.rs"]
mod lifecycle;
#[path = "gate_actuators/numerical.rs"]
mod numerical;
#[path = "gate_actuators/previous_completion.rs"]
mod previous_completion;
#[path = "../examples/support/gate_actuators/context.rs"]
mod reference_context;
#[path = "gate_actuators/restoration.rs"]
mod restoration;
