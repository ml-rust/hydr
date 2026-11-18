//! Free delivery outlet that bounds the lift from below.

use crate::structures::{StructureError, StructureInputErrorKind};

/// Delivery outlet that discharges to atmosphere at a fixed common-datum elevation.
///
/// A running pump lifts to `max(receiving stage, elevation_m)`. A dry receiver whose bed lies at
/// or below the outlet then takes delivery.
#[derive(Clone, Debug, PartialEq)]
pub struct PumpFreeOutlet {
    /// Common-datum elevation at which delivery leaves the pipe at atmospheric pressure, in metres.
    pub elevation_m: f64,
    /// Evidence that the outlet discharges freely at this elevation and that the curve's head
    /// argument holds there.
    pub evidence: String,
}
impl PumpFreeOutlet {
    pub(crate) fn check(&self) -> Result<(), StructureError> {
        if !self.elevation_m.is_finite() {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!(
                    "free outlet elevation_m={}: supply a finite common-datum outlet elevation",
                    self.elevation_m
                ),
            ));
        }
        crate::structures::definition::text(&self.evidence, "free outlet evidence")
    }
}
