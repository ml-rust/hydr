use super::{Domain, ModelError};

/// Depth and depth-integrated horizontal momentum for one Cartesian cell.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CellState {
    /// Water depth in metres.
    pub h: f64,
    /// Depth times x velocity in square metres per second.
    pub qx: f64,
    /// Depth times y velocity in square metres per second.
    pub qy: f64,
}

/// Checked, owned row-major flow state.
#[derive(Clone, Debug, PartialEq)]
pub struct FlowState {
    cells: Vec<CellState>,
}

impl FlowState {
    /// Creates flow state compatible with the supplied domain.
    ///
    /// Positive films remain positive. Dry and inactive cells require zero momentum.
    ///
    /// # Errors
    ///
    /// Rejects mismatched shape, invalid depths or momenta, unsupported-cell water, and unrepresentable derived quantities.
    pub fn new(domain: &Domain, cells: Vec<CellState>) -> Result<Self, ModelError> {
        let count = domain.grid().cell_count();
        if cells.len() != count {
            return Err(ModelError::invalid(
                "cells",
                None,
                format!(
                    "received {} cells: supply exactly {count} row-major states",
                    cells.len()
                ),
            ));
        }
        let area = domain.grid().cell_area();
        let mut total_volume = 0.0;
        for (index, ((cell, active), bed)) in cells
            .iter()
            .zip(domain.active())
            .zip(domain.bed())
            .enumerate()
        {
            if !cell.h.is_finite() || cell.h < 0.0 {
                return Err(ModelError::invalid(
                    "h",
                    Some(index),
                    format!(
                        "depth {} is invalid: supply finite nonnegative metres",
                        cell.h
                    ),
                ));
            }
            for (field, momentum) in [("qx", cell.qx), ("qy", cell.qy)] {
                if !momentum.is_finite() {
                    return Err(ModelError::invalid(
                        field,
                        Some(index),
                        format!(
                            "momentum {momentum} is nonfinite: supply finite square metres per second"
                        ),
                    ));
                }
                if cell.h == 0.0 && momentum != 0.0 {
                    return Err(ModelError::invalid(
                        field,
                        Some(index),
                        format!("dry cell carries momentum {momentum}: supply zero momentum"),
                    ));
                }
                if cell.h > 0.0 && !(momentum / cell.h).is_finite() {
                    return Err(ModelError::invalid(
                        field,
                        Some(index),
                        format!(
                            "momentum {momentum} divided by depth {} overflows: reduce velocity",
                            cell.h
                        ),
                    ));
                }
            }
            if !active && *cell != CellState::default() {
                return Err(ModelError::invalid(
                    "cells",
                    Some(index),
                    format!("inactive cell contains {cell:?}: supply zero depth and momentum"),
                ));
            }
            if !(bed + cell.h).is_finite() {
                return Err(ModelError::invalid(
                    "h",
                    Some(index),
                    format!(
                        "bed {bed} plus depth {} overflows: reduce elevation or depth",
                        cell.h
                    ),
                ));
            }
            let volume = cell.h * area;
            if !volume.is_finite() || (cell.h > 0.0 && volume == 0.0) {
                return Err(ModelError::invalid(
                    "h",
                    Some(index),
                    format!(
                        "depth {} times cell area {area} is unrepresentable: rescale the domain or depth",
                        cell.h
                    ),
                ));
            }
            total_volume += volume;
            if !total_volume.is_finite() {
                return Err(ModelError::invalid(
                    "cells",
                    Some(index),
                    "total water volume overflows: reduce the domain or initial storage",
                ));
            }
        }
        Ok(Self { cells })
    }

    /// Returns row-major cell states in SI units.
    pub fn cells(&self) -> &[CellState] {
        &self.cells
    }
}
