//! One outlet-control row: barrel, length, roughness, entrance loss, slope, gravity and the
//! entrance face blockage.

use crate::structures::{BlockageFraction, CulvertBarrel, StructureError};

/// Barrel, length, Manning `n`, entrance loss, slope, gravity and entrance face blockage of one
/// outlet-control row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CulvertOutletRow {
    /// Barrel the flow runs through. A barrel with a floor runs its profile on the open section,
    /// with depths, tailwater and headwater above the bed.
    pub barrel: CulvertBarrel,
    /// Barrel length `L` in metres.
    pub length_m: f64,
    /// Manning roughness `n` in `s/m^(1/3)`. Above a floor it is the wall `n`, and friction
    /// takes the weighted perimeter `P_n = P_w + rho^(3/2) b`.
    pub manning_n: f64,
    /// Entrance loss coefficient `ke` of the inlet that the flow enters.
    pub entrance_loss: f64,
    /// Signed barrel slope `S` in metres per metre, positive when the barrel falls from the inlet
    /// to the outlet. The inlet invert sits `S L` above the outlet invert.
    pub slope: f64,
    /// Gravity in metres per second squared.
    pub gravity_m_per_s2: f64,
    /// Blocked share `f` of the entrance face. The entrance loss becomes
    /// `ke / (1 - f)^2 + (f / (1 - f))^2`: the entrance loss at the face velocity `V / (1 - f)`
    /// plus the expansion back to `V`. This rule is a hydr definition.
    pub face_blockage: BlockageFraction,
}

impl CulvertOutletRow {
    /// Rejects a nonfinite or nonpositive length, roughness or gravity, a nonfinite or negative
    /// entrance loss, a nonfinite slope, and an adverse barrel whose outlet sits more than `D/2`
    /// above its inlet, with `D'` above a floor.
    pub(crate) fn check(&self) -> Result<(), StructureError> {
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
        let rise = self.barrel.open_rise_m();
        if self.slope * self.length_m < -0.5 * rise {
            let bed = match self.barrel.floor() {
                Some(floor) => format!(",sediment_depth_m={:?}", floor.sediment_depth_m),
                None => String::new(),
            };
            return Err(StructureError::definition(format!(
                "{fields},rise_m={rise:?}{bed}: the outlet invert sits more than half the rise above the inlet invert. Supply an adverse fall of at most rise/2"
            )));
        }
        Ok(())
    }
}
