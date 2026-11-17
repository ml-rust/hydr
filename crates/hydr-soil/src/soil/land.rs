//! Explicit land support and complete antecedent memory.

use super::{ModifiedGreenAmptParameters, NativeSoilState, SoilError};
use crate::model::Domain;

/// Physical land model for one hydraulic cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LandCell {
    /// No soil infiltration mechanism.
    Impervious,
    /// Pinned Modified Green–Ampt soil with checked parameters.
    ModifiedGreenAmpt(ModifiedGreenAmptParameters),
}

/// Checked row-major land support. Inactive hydraulic cells require impervious bookkeeping.
#[derive(Clone, Debug, PartialEq)]
pub struct LandModel {
    cells: Vec<LandCell>,
}

impl LandModel {
    /// Check physical land support against the hydraulic domain.
    pub fn new(domain: &Domain, cells: Vec<LandCell>) -> Result<Self, SoilError> {
        if cells.len() != domain.grid().cell_count() {
            return Err(SoilError::input(format!(
                "land cells={}: require {}",
                cells.len(),
                domain.grid().cell_count()
            )));
        }
        for (cell, (&land, &active)) in cells.iter().zip(domain.active()).enumerate() {
            if !active && land != LandCell::Impervious {
                return Err(SoilError::InvalidInput {
                    cell: Some(cell),
                    context: "inactive cells require explicit impervious bookkeeping".into(),
                });
            }
        }
        Ok(Self { cells })
    }
    /// Explicitly select impervious land for every domain cell.
    pub fn impervious(domain: &Domain) -> Self {
        Self {
            cells: vec![LandCell::Impervious; domain.grid().cell_count()],
        }
    }
    /// Immutable row-major physical land definitions.
    pub fn cells(&self) -> &[LandCell] {
        &self.cells
    }
}

/// Explicit antecedent memory matching one land cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SoilCellState {
    /// No native soil state for impervious land.
    Impervious,
    /// Complete native memory for a Modified Green–Ampt cell.
    ModifiedGreenAmpt(NativeSoilState),
}

/// Checked full-grid antecedent memory with exact land correspondence.
#[derive(Clone, Debug, PartialEq)]
pub struct SoilState {
    cells: Vec<SoilCellState>,
}

impl SoilState {
    /// Check every antecedent cell against its explicit physical land model.
    pub fn new(land: &LandModel, cells: Vec<SoilCellState>) -> Result<Self, SoilError> {
        if cells.len() != land.cells.len() {
            return Err(SoilError::input(format!(
                "soil cells={}: require {}",
                cells.len(),
                land.cells.len()
            )));
        }
        for (index, (model, state)) in land.cells.iter().zip(&cells).enumerate() {
            match (model, state) {
                (LandCell::Impervious, SoilCellState::Impervious) => {}
                (
                    LandCell::ModifiedGreenAmpt(parameters),
                    SoilCellState::ModifiedGreenAmpt(state),
                ) => state
                    .check(*parameters, index)
                    .map_err(|error| error.at_cell(index))?,
                _ => {
                    return Err(SoilError::InvalidInput {
                        cell: Some(index),
                        context: "soil variant must match the explicit land variant".into(),
                    });
                }
            }
        }
        Ok(Self { cells })
    }
    /// Explicitly select original EPA initialization for every pervious cell.
    pub fn initialized(land: &LandModel) -> Result<Self, SoilError> {
        let cells = land
            .cells
            .iter()
            .map(|model| match model {
                LandCell::Impervious => SoilCellState::Impervious,
                LandCell::ModifiedGreenAmpt(parameters) => {
                    SoilCellState::ModifiedGreenAmpt(NativeSoilState {
                        imd: parameters.maximum_deficit(),
                        f_ft: 0.0,
                        fu_ft: 0.0,
                        saturated: false,
                        t_s: 0.0,
                    })
                }
            })
            .collect();
        Self::new(land, cells)
    }
    /// Immutable complete row-major antecedent memory.
    pub fn cells(&self) -> &[SoilCellState] {
        &self.cells
    }
}
