use super::{fixture, reference_case};

#[path = "admission/assertions.rs"]
mod assertions;
#[path = "admission/definitions.rs"]
mod definitions;
#[path = "admission/initial.rs"]
mod initial;
#[path = "admission/samples.rs"]
mod samples;

use assertions::rejects;
