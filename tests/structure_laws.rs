//! Typed hydraulic support, independent constitutive references, and immutable proposals.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "structure_laws/admission.rs"]
mod admission;
#[path = "structure_laws/behavior.rs"]
mod behavior;
#[path = "structure_laws/fixture.rs"]
mod fixture;
#[path = "structure_laws/held.rs"]
mod held;
#[path = "structure_laws/references.rs"]
mod references;
