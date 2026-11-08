use crate::error::GridError;

/// Uniform Cartesian grid with SI spacing and no allocated flow state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartesianGrid {
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
    cell_count: usize,
}

impl CartesianGrid {
    /// Creates grid geometry with `nx` columns and `ny` rows.
    ///
    /// Spacing uses metres. Construction allocates no cell storage.
    ///
    /// # Errors
    ///
    /// Rejects empty dimensions, overflowing cell counts, and nonfinite or nonpositive geometry.
    pub fn new(nx: usize, ny: usize, dx: f64, dy: f64) -> Result<Self, GridError> {
        if nx == 0 || ny == 0 {
            return Err(GridError::EmptyDimensions { nx, ny });
        }
        let cell_count = nx
            .checked_mul(ny)
            .ok_or(GridError::CellCountOverflow { nx, ny })?;
        let geometry = [dx, dy, dx * dy, nx as f64 * dx, ny as f64 * dy];
        if geometry
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(GridError::InvalidSpacing { dx, dy });
        }
        Ok(Self {
            nx,
            ny,
            dx,
            dy,
            cell_count,
        })
    }

    /// Returns the column and row counts, respectively.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// Returns cell width and height in metres, respectively.
    pub fn spacing(&self) -> (f64, f64) {
        (self.dx, self.dy)
    }

    /// Returns the number of cells.
    pub fn cell_count(&self) -> usize {
        self.cell_count
    }

    /// Returns cell area in square metres.
    pub fn cell_area(&self) -> f64 {
        self.dx * self.dy
    }

    /// Returns domain width and height in metres, respectively.
    pub fn extent(&self) -> (f64, f64) {
        (self.nx as f64 * self.dx, self.ny as f64 * self.dy)
    }
}
