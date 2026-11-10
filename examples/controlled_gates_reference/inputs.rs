//! Original control, response, and accepted numerical policy bits.
#[path = "endpoint_inputs.rs"]
mod endpoint_inputs;
#[path = "input_catalog.rs"]
mod input_catalog;
#[path = "input_types.rs"]
mod input_types;
#[path = "motion_inputs.rs"]
mod motion_inputs;
#[path = "rejection_inputs.rs"]
mod rejection_inputs;
pub use input_catalog::cases;
pub use input_types::{Case, Decision, Query};
