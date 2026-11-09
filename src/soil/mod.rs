//! Immutable batched Modified Green–Ampt soil on the numr CPU runtime.

mod batch;
mod error;
mod parameters;
mod state;

pub use batch::{ModifiedGreenAmptBatch, ModifiedGreenAmptTrial};
pub use error::SoilError;
pub use parameters::ModifiedGreenAmptParameters;
pub use state::{NativeSoilInput, NativeSoilState, SoilInput, SoilOutput};
