use crate::grid::CartesianGrid;

use super::{ModelError, SpatialReference};

/// Immutable Cartesian terrain, roughness, mask, and reference metadata.
///
/// Arrays follow row-major grid order. Inactive cells declare impermeable exclusions and return unsupported outputs.
/// Unknown influential terrain cannot become an inactive-cell wall without physical evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct Domain {
    grid: CartesianGrid,
    bed: Vec<f64>,
    manning: Vec<f64>,
    active: Vec<bool>,
    reference: SpatialReference,
}

impl Domain {
    /// Creates a checked hydraulic domain.
    ///
    /// Bed elevations use metres. Manning coefficients use seconds per metre to the one-third power.
    /// Inactive elevations accept unknown values and normalize to zero for exact snapshot equality.
    ///
    /// # Errors
    ///
    /// Rejects mismatched arrays, an empty active mask, nonfinite active beds, and invalid Manning coefficients.
    pub fn new(
        grid: CartesianGrid,
        mut bed: Vec<f64>,
        manning: Vec<f64>,
        active: Vec<bool>,
        reference: SpatialReference,
    ) -> Result<Self, ModelError> {
        let count = grid.cell_count();
        let origin = reference.origin();
        let extent = grid.extent();
        for (axis, length) in [extent.0, extent.1].into_iter().enumerate() {
            let end = origin[axis] + length;
            if !end.is_finite() || end <= origin[axis] {
                return Err(ModelError::invalid(
                    "origin",
                    Some(axis),
                    format!(
                        "origin={} plus extent={length} is unrepresentable: rescale projected coordinates",
                        origin[axis]
                    ),
                ));
            }
        }
        for (field, length) in [
            ("bed", bed.len()),
            ("manning", manning.len()),
            ("active", active.len()),
        ] {
            if length != count {
                return Err(ModelError::invalid(
                    field,
                    None,
                    format!("received {length} cells: supply exactly {count} row-major values"),
                ));
            }
        }
        if !active.iter().any(|&value| value) {
            return Err(ModelError::invalid(
                "active",
                None,
                "mask contains no active cells: mark at least one supported cell active",
            ));
        }
        for (index, ((elevation, coefficient), enabled)) in
            bed.iter_mut().zip(&manning).zip(&active).enumerate()
        {
            if *enabled && !elevation.is_finite() {
                return Err(ModelError::invalid(
                    "bed",
                    Some(index),
                    format!(
                        "elevation {elevation} is nonfinite: supply finite metres for active cells"
                    ),
                ));
            }
            if !coefficient.is_finite() || *coefficient < 0.0 {
                return Err(ModelError::invalid(
                    "manning",
                    Some(index),
                    format!(
                        "coefficient {coefficient} is invalid: supply a finite nonnegative coefficient"
                    ),
                ));
            }
            if !enabled {
                *elevation = 0.0;
            }
        }
        Ok(Self {
            grid,
            bed,
            manning,
            active,
            reference,
        })
    }

    /// Returns the Cartesian geometry.
    pub fn grid(&self) -> CartesianGrid {
        self.grid
    }

    /// Returns row-major bed elevations in metres, with inactive elevations normalized to zero.
    pub fn bed(&self) -> &[f64] {
        &self.bed
    }

    /// Returns row-major Manning coefficients in SI units.
    pub fn manning(&self) -> &[f64] {
        &self.manning
    }

    /// Returns the row-major supported-cell mask.
    pub fn active(&self) -> &[bool] {
        &self.active
    }

    /// Returns spatial reference metadata.
    pub fn reference(&self) -> &SpatialReference {
        &self.reference
    }
}
