//! Immutable batched Modified Green–Ampt soil on the numr CPU runtime.

pub(crate) mod admission;
mod batch;
mod continuation;
mod error;
mod land;
mod parameters;
mod receipt;
mod state;
mod surface_trial;

pub use batch::{ModifiedGreenAmptBatch, ModifiedGreenAmptTrial};
pub use error::SoilError;
pub use land::{LandCell, LandModel, SoilCellState, SoilState};
pub use parameters::ModifiedGreenAmptParameters;
pub use receipt::SoilSourceReceipt;
pub(crate) use receipt::{SoilReceiptInput, check_source_receipt};
pub use state::{NativeSoilInput, NativeSoilState, SoilInput, SoilOutput};
