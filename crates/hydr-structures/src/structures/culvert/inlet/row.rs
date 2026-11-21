//! One inlet-control row: barrel, inlet, signed barrel slope and gravity.

use super::CulvertInlet;
use crate::structures::{CulvertBarrel, StructureError};

/// Barrel, inlet, slope and gravity of one inlet-control row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertInletRow {
    /// Barrel the inlet leads into.
    pub barrel: CulvertBarrel,
    /// HDS-5 inlet definition.
    pub inlet: CulvertInlet,
    /// Signed barrel slope `S` in metres per metre, positive when the barrel falls away from the
    /// inlet. It enters the slope term `Ks S`.
    pub slope: f64,
    /// Gravity in metres per second squared, for the form-1 critical head.
    pub gravity_m_per_s2: f64,
}

impl CulvertInletRow {
    /// Rejects a nonfinite slope, a nonfinite or nonpositive gravity, and a coded inlet on a
    /// barrel of another shape.
    pub(super) fn check(&self) -> Result<(), StructureError> {
        let inlet = self.inlet.describe();
        if !self.slope.is_finite() {
            return Err(StructureError::definition(format!(
                "{inlet},slope={:?}: supply a finite barrel slope in metres per metre",
                self.slope
            )));
        }
        let gravity = self.gravity_m_per_s2;
        if !(gravity.is_finite() && gravity > 0.0) {
            return Err(StructureError::definition(format!(
                "{inlet},gravity_m_per_s2={gravity:?}: supply a finite positive gravity in metres per second squared"
            )));
        }
        if let Some(shape) = self.inlet.barrel_shape()
            && shape != self.barrel.shape()
        {
            return Err(StructureError::definition(format!(
                "{inlet},barrel_shape={:?}: the code is for {shape:?} barrels. Supply a code for the barrel shape or explicit coefficients",
                self.barrel.shape()
            )));
        }
        Ok(())
    }
}
