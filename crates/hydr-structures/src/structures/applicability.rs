//! Explicit coefficient and physical-regime evidence without inferred defaults.

/// Applicability evidence for a fully submerged opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrificeApplicability {
    /// Coefficient source, calibration, or declared assumption.
    pub coefficient_provenance: String,
    /// Evidence supporting the fully submerged law and quiescent approach assumptions.
    pub fully_submerged_applicability: String,
    /// Explicit coefficient applicability under reversed flow.
    pub symmetric_reversal: bool,
}
impl OrificeApplicability {
    pub(crate) fn check(&self) -> Result<(), super::StructureError> {
        super::definition::text(&self.coefficient_provenance, "coefficient_provenance")?;
        super::definition::text(
            &self.fully_submerged_applicability,
            "fully_submerged_applicability",
        )
    }
}

/// Applicability evidence for an HDS-5 culvert.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CulvertApplicability {
    /// Source of the inlet coefficients, entrance losses and Manning `n`.
    pub coefficient_provenance: String,
    /// Evidence that the barrel is clean and unobstructed, and that its stored water is negligible.
    pub barrel_condition_assertion: String,
}
impl CulvertApplicability {
    pub(crate) fn check(&self) -> Result<(), super::StructureError> {
        super::definition::text(&self.coefficient_provenance, "coefficient_provenance")?;
        super::definition::text(
            &self.barrel_condition_assertion,
            "barrel_condition_assertion",
        )
    }
}

/// Applicability evidence for a suppressed sharp-crested rectangular weir only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeirApplicability {
    /// Coefficient source, calibration, or declared assumption.
    pub coefficient_provenance: String,
    /// Explicit suppressed-crest, free-nappe, and approach-flow assumptions.
    pub free_nappe_and_approach_assertion: String,
    /// Applicability evidence for the adopted Villemonte correction.
    pub submergence_correction_assertion: String,
    /// Explicit coefficient applicability under reversed flow.
    pub symmetric_reversal: bool,
}
impl WeirApplicability {
    pub(crate) fn check(&self) -> Result<(), super::StructureError> {
        for (value, field) in [
            (&self.coefficient_provenance, "coefficient_provenance"),
            (
                &self.free_nappe_and_approach_assertion,
                "free_nappe_and_approach_assertion",
            ),
            (
                &self.submergence_correction_assertion,
                "submergence_correction_assertion",
            ),
        ] {
            super::definition::text(value, field)?;
        }
        Ok(())
    }
}
