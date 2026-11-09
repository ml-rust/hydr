//! Explicit coefficient and hydraulic-envelope evidence for rectangular gates.

use crate::structures::{StructureError, definition};

/// Evidence for the hydrostatic pressure-strip closure across the admitted aperture envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateApplicability {
    /// Coefficient source, calibration, or declared assumption.
    pub coefficient_provenance: String,
    /// Evidence for hydrostatic pressures and negligible approach velocity between quiescent pools.
    pub hydrostatic_quiescent_pool_assertion: String,
    /// Coefficient applicability across free, mixed, submerged, and partly wetted openings.
    pub free_submerged_aperture_applicability: String,
    /// Evidence that both connections expose the complete sill-to-opening aperture.
    pub unobstructed_aperture_assertion: String,
    /// Explicit coefficient applicability under reversed flow.
    pub symmetric_reversal: bool,
}
impl GateApplicability {
    pub(crate) fn check(&self) -> Result<(), StructureError> {
        for (value, field) in [
            (&self.coefficient_provenance, "coefficient_provenance"),
            (
                &self.hydrostatic_quiescent_pool_assertion,
                "hydrostatic_quiescent_pool_assertion",
            ),
            (
                &self.free_submerged_aperture_applicability,
                "free_submerged_aperture_applicability",
            ),
            (
                &self.unobstructed_aperture_assertion,
                "unobstructed_aperture_assertion",
            ),
        ] {
            definition::text(value, field)?;
        }
        Ok(())
    }
}
