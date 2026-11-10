//! Original input bits and explicit numerical policies.
#[path = "command_inputs.rs"]
mod command_inputs;
#[path = "extreme_inputs.rs"]
mod extreme_inputs;
#[path = "input_catalog.rs"]
mod input_catalog;
#[path = "input_types.rs"]
mod input_types;
#[path = "ordinary_inputs.rs"]
mod ordinary_inputs;
#[path = "small_inputs.rs"]
mod small_inputs;
#[path = "supplement_inputs.rs"]
mod supplement_inputs;
pub use input_catalog::cases;
pub use input_types::{Case, Query, Update};
