//! Analytical geometry, profile admission, and ragged batching checks.

#[path = "cross_sections/admission.rs"]
mod admission;
#[path = "cross_sections/analytical.rs"]
mod analytical;
#[path = "cross_sections/layouts.rs"]
mod layouts;
#[path = "cross_sections/profiles.rs"]
mod profiles;
#[path = "cross_sections/recovery.rs"]
mod recovery;
#[path = "support/runtime.rs"]
mod runtime;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
type Batch = hydr::channel::CrossSectionBatch<runtime::Runtime>;
