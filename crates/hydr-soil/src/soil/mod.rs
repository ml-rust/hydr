//! Immutable batched Modified Green–Ampt soil on the numr CPU runtime.

mod batch;
mod capabilities;
mod continuation;
mod error;
#[doc(hidden)]
pub mod kernel;
mod land;
mod parameters;
mod receipt;
mod state;
mod surface_trial;

pub use batch::{ModifiedGreenAmptBatch, ModifiedGreenAmptTrial};
pub use capabilities::soil_capabilities;
pub use error::SoilError;
pub use land::{LandCell, LandModel, SoilCellState, SoilState};
pub use parameters::ModifiedGreenAmptParameters;
pub use receipt::SoilSourceReceipt;
#[doc(hidden)]
pub use receipt::{SoilReceiptInput, check_source_receipt};
pub use state::{NativeSoilInput, NativeSoilState, SoilInput, SoilOutput};
#[doc(hidden)]
pub use surface_trial::SoilSurfaceTrial;
