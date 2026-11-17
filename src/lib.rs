//! Foundations for hydraulic simulation in Rust.
//!
//! Configured Numr runtimes advance full 2D shallow-water flow through atomic accepted steps.
//! CPU F64 is the checked reference profile. Other runtimes must satisfy each algorithm's arithmetic requirements.
//! Read [`solver::capabilities`] before selecting application boundaries and physical regimes.
//!
//! ```
//! use hydr::grid::CartesianGrid;
//!
//! let grid = CartesianGrid::new(100, 50, 2.0, 2.0)?;
//! assert_eq!(grid.cell_count(), 5_000);
//! # Ok::<(), hydr::error::GridError>(())
//! ```

#[doc(inline)]
pub use hydr_core::{error, forcing, grid, model};

use hydr_core::{contract, numerics, tensor};

#[doc(inline)]
pub use hydr_channel::channel;
pub mod coupling;
#[doc(inline)]
pub use hydr_runoff::runoff;
#[doc(inline)]
pub use hydr_soil::soil;

pub mod solver;
#[doc(inline)]
pub use hydr_storage::storage;

#[doc(inline)]
pub use hydr_structures::structures;
