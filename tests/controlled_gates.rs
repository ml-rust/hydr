//! Atomic sampled commands, actual gate motion, and complete paired continuation.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "controlled_gate_support/admission.rs"]
mod admission;
#[path = "controlled_gate_support/fixture.rs"]
mod fixture;
#[path = "controlled_gate_support/motion.rs"]
mod motion;
#[path = "controlled_gate_support/restoration.rs"]
mod restoration;
