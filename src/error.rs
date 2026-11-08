//! Errors for grid construction.

use std::fmt;

/// Invalid Cartesian grid geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum GridError {
    /// At least one dimension contains zero cells.
    EmptyDimensions {
        /// Number of columns.
        nx: usize,
        /// Number of rows.
        ny: usize,
    },
    /// Cell count exceeds the platform's address space.
    CellCountOverflow {
        /// Number of columns.
        nx: usize,
        /// Number of rows.
        ny: usize,
    },
    /// Spacing or derived geometry is nonfinite or nonpositive.
    InvalidSpacing {
        /// Cell width in metres.
        dx: f64,
        /// Cell height in metres.
        dy: f64,
    },
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDimensions { nx, ny } => write!(
                f,
                "invalid grid dimensions nx={nx}, ny={ny}: use nonzero dimensions"
            ),
            Self::CellCountOverflow { nx, ny } => write!(
                f,
                "grid cell count overflows for nx={nx}, ny={ny}: reduce dimensions"
            ),
            Self::InvalidSpacing { dx, dy } => write!(
                f,
                "invalid grid spacing dx={dx}, dy={dy}: use finite positive spacing with representable area and extents"
            ),
        }
    }
}

impl std::error::Error for GridError {}
