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

pub mod error;
pub mod grid;
pub mod model;
mod numerics;
pub mod runoff;
pub mod soil;
pub mod solver;
pub mod storage;
pub mod structures;
mod tensor;
