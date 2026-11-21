//! One outlet-control row: barrel, length, roughness, entrance loss, slope and gravity.

use crate::structures::{CulvertBarrel, StructureError};

/// Barrel, length, Manning `n`, entrance loss, slope and gravity of one outlet-control row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertOutletRow {
    /// Barrel the flow runs through.
    pub barrel: CulvertBarrel,
    /// Barrel length `L` in metres.
    pub length_m: f64,
    /// Manning roughness `n` in `s/m^(1/3)`.
    pub manning_n: f64,
    /// Entrance loss coefficient `ke` of the inlet that the flow enters.
    pub entrance_loss: f64,
    /// Signed barrel slope `S` in metres per metre, positive when the barrel falls from the inlet
    /// to the outlet. The inlet invert sits `S L` above the outlet invert.
    pub slope: f64,
    /// Gravity in metres per second squared.
    pub gravity_m_per_s2: f64,
}

impl CulvertOutletRow {
    /// Rejects a nonfinite or nonpositive length, roughness or gravity, a nonfinite or negative
    /// entrance loss, a nonfinite slope, and an adverse barrel whose outlet invert sits more than
    /// `D/2` above its inlet invert.
    pub(super) fn check(&self) -> Result<(), StructureError> {
        let fields = format!(
            "length_m={:?},manning_n={:?},entrance_loss={:?},slope={:?},gravity_m_per_s2={:?}",
            self.length_m, self.manning_n, self.entrance_loss, self.slope, self.gravity_m_per_s2
        );
        let positive = [
            (self.length_m, "barrel length in metres"),
            (self.manning_n, "Manning n in s/m^(1/3)"),
            (
                self.gravity_m_per_s2,
                "gravity in metres per second squared",
            ),
        ];
        for (value, what) in positive {
            if !(value.is_finite() && value > 0.0) {
                return Err(StructureError::definition(format!(
                    "{fields}: supply a finite positive {what}"
                )));
            }
        }
        if !(self.entrance_loss.is_finite() && self.entrance_loss >= 0.0) {
            return Err(StructureError::definition(format!(
                "{fields}: supply a finite nonnegative entrance loss coefficient"
            )));
        }
        if !self.slope.is_finite() {
            return Err(StructureError::definition(format!(
                "{fields}: supply a finite barrel slope in metres per metre"
            )));
        }
        let rise = self.barrel.rise_m();
        if self.slope * self.length_m < -0.5 * rise {
            return Err(StructureError::definition(format!(
                "{fields},rise_m={rise:?}: the outlet invert sits more than half the rise above the inlet invert. Supply an adverse fall of at most rise/2"
            )));
        }
        Ok(())
    }
}
