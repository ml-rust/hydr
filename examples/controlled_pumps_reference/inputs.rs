//! Frozen original command, curve-head probe, and calendar inputs.
#[path = "admission_inputs.rs"]
mod admission_inputs;
#[path = "adoption_inputs.rs"]
mod adoption_inputs;
#[path = "input_catalog.rs"]
mod input_catalog;
#[path = "input_types.rs"]
mod input_types;
#[path = "qualification_inputs.rs"]
mod qualification_inputs;
#[path = "switch_inputs.rs"]
mod switch_inputs;
pub use admission_inputs::admissions;
pub use input_catalog::cases;
pub use input_types::{Admission, Case, Decision, Support};
