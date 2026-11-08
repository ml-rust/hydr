//! Checked hydraulic domains, SI flow state, and spatial reference metadata.

mod domain;
mod error;
mod reference;
mod state;

pub use domain::Domain;
pub use error::ModelError;
pub use reference::SpatialReference;
pub use state::{CellState, FlowState};
