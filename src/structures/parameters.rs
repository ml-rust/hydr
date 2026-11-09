//! Typed immutable geometry and coefficient records, without coefficient-product admission.

use super::{OrificeApplicability, StructureError, StructureInputErrorKind, WeirApplicability};
/// Complete checked SubmergedOrificeParameters geometry and applicability.
#[derive(Clone, Debug, PartialEq)]
pub struct SubmergedOrificeParameters {
    area_m2: f64,
    sill_m: f64,
    crown_m: f64,
    discharge_coefficient: f64,
    gravity_m_per_s2: f64,
    applicability: OrificeApplicability,
}
impl SubmergedOrificeParameters {
    /// Check finite geometry, positive coefficients, and complete applicability evidence.
    pub fn new(
        area_m2: f64,
        sill_m: f64,
        crown_m: f64,
        discharge_coefficient: f64,
        gravity_m_per_s2: f64,
        applicability: OrificeApplicability,
    ) -> Result<Self, StructureError> {
        applicability.check()?;
        if [
            area_m2,
            sill_m,
            crown_m,
            discharge_coefficient,
            gravity_m_per_s2,
        ]
        .iter()
        .any(|v| !v.is_finite())
            || [area_m2, discharge_coefficient, gravity_m_per_s2]
                .iter()
                .any(|v| *v <= 0.0)
            || crown_m <= sill_m
        {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!(
                    "invalid SubmergedOrificeParameters: area_m2={area_m2},sill_m={sill_m},crown_m={crown_m},discharge_coefficient={discharge_coefficient},gravity_m_per_s2={gravity_m_per_s2}; supply finite elevations with crown above sill and positive area/coefficient/gravity"
                ),
            ));
        }
        Ok(Self {
            area_m2,
            sill_m,
            crown_m,
            discharge_coefficient,
            gravity_m_per_s2,
            applicability,
        })
    }
    /// Opening area in m².
    pub fn area_m2(&self) -> f64 {
        self.area_m2
    }
    /// Opening sill elevation on the common datum.
    pub fn sill_m(&self) -> f64 {
        self.sill_m
    }
    /// Opening crown elevation on the common datum.
    pub fn crown_m(&self) -> f64 {
        self.crown_m
    }
    /// Explicit positive dimensionless discharge coefficient.
    pub fn discharge_coefficient(&self) -> f64 {
        self.discharge_coefficient
    }
    /// Explicit positive gravity in m/s².
    pub fn gravity_m_per_s2(&self) -> f64 {
        self.gravity_m_per_s2
    }
    /// Complete immutable applicability evidence.
    pub fn applicability(&self) -> &OrificeApplicability {
        &self.applicability
    }
}
/// Complete checked SuppressedRectangularWeirParameters geometry and applicability.
#[derive(Clone, Debug, PartialEq)]
pub struct SuppressedRectangularWeirParameters {
    crest_length_m: f64,
    crest_m: f64,
    discharge_coefficient: f64,
    applicability: WeirApplicability,
}
impl SuppressedRectangularWeirParameters {
    /// Check finite geometry, positive coefficients, and complete applicability evidence.
    pub fn new(
        crest_length_m: f64,
        crest_m: f64,
        discharge_coefficient: f64,
        applicability: WeirApplicability,
    ) -> Result<Self, StructureError> {
        applicability.check()?;
        if [crest_length_m, crest_m, discharge_coefficient]
            .iter()
            .any(|v| !v.is_finite())
            || [crest_length_m, discharge_coefficient]
                .iter()
                .any(|v| *v <= 0.0)
        {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!(
                    "invalid SuppressedRectangularWeirParameters: crest_length_m={crest_length_m},crest_m={crest_m},discharge_coefficient={discharge_coefficient}; supply finite crest elevation and positive crest length/coefficient"
                ),
            ));
        }
        Ok(Self {
            crest_length_m,
            crest_m,
            discharge_coefficient,
            applicability,
        })
    }
    /// Suppressed sharp-crest length in metres.
    pub fn crest_length_m(&self) -> f64 {
        self.crest_length_m
    }
    /// Crest elevation on the common datum.
    pub fn crest_m(&self) -> f64 {
        self.crest_m
    }
    /// Explicit positive coefficient in SI m^(1/2)/s.
    pub fn discharge_coefficient(&self) -> f64 {
        self.discharge_coefficient
    }
    /// Complete immutable applicability evidence.
    pub fn applicability(&self) -> &WeirApplicability {
        &self.applicability
    }
}
