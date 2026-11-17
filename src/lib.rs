//! Foundations for hydraulic simulation in Rust.
//!
//! Configured Numr runtimes advance full 2D shallow-water flow through atomic accepted steps.
//! CPU F64 is the checked reference profile. Other runtimes must satisfy each algorithm's arithmetic requirements.
//! Read [`solver::surface_solver_capabilities`] before selecting application boundaries and physical regimes.
//! It returns a [`capability::ComponentCapabilities`] declaration.
//!
//! ```
//! use hydr::grid::CartesianGrid;
//!
//! let grid = CartesianGrid::new(100, 50, 2.0, 2.0)?;
//! assert_eq!(grid.cell_count(), 5_000);
//! # Ok::<(), hydr::error::GridError>(())
//! ```

#[doc(inline)]
pub use hydr_core::{capability, error, forcing, grid, model};

#[doc(inline)]
pub use hydr_channel::channel;
#[doc(inline)]
pub use hydr_coupling::coupling;
#[doc(inline)]
pub use hydr_runoff::runoff;
#[doc(inline)]
pub use hydr_soil::soil;

#[doc(inline)]
pub use hydr_solver::solver;
#[doc(inline)]
pub use hydr_storage::storage;

#[doc(inline)]
pub use hydr_structures::structures;
