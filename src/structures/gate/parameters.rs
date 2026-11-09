//! Immutable rectangular-aperture geometry and common-regime coefficient.

use super::GateApplicability;
use crate::structures::{StructureError, StructureInputErrorKind};

/// Thin vertical rectangular aperture between explicitly quiescent hydrostatic pools.
#[derive(Clone, Debug, PartialEq)]
pub struct GateParameters {
    width_m: f64,
    sill_m: f64,
    maximum_opening_m: f64,
    discharge_coefficient: f64,
    gravity_m_per_s2: f64,
    applicability: GateApplicability,
}
impl GateParameters {
    /// Check finite sill, positive geometry and coefficients, and complete applicability evidence.
    pub fn new(
        width_m: f64,
        sill_m: f64,
        maximum_opening_m: f64,
        discharge_coefficient: f64,
        gravity_m_per_s2: f64,
        applicability: GateApplicability,
    ) -> Result<Self, StructureError> {
        applicability.check()?;
        if !sill_m.is_finite()
            || [
                width_m,
                maximum_opening_m,
                discharge_coefficient,
                gravity_m_per_s2,
            ]
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!(
                    "width_m={width_m},sill_m={sill_m},maximum_opening_m={maximum_opening_m},discharge_coefficient={discharge_coefficient},gravity_m_per_s2={gravity_m_per_s2}: supply finite sill and positive finite width, maximum opening, coefficient, and gravity"
                ),
            ));
        }
        Ok(Self {
            width_m,
            sill_m,
            maximum_opening_m,
            discharge_coefficient,
            gravity_m_per_s2,
            applicability,
        })
    }
    /// Actual aperture width before contraction, in metres.
    pub fn width_m(&self) -> f64 {
        self.width_m
    }
    /// Aperture sill elevation on the declared common datum, in metres.
    pub fn sill_m(&self) -> f64 {
        self.sill_m
    }
    /// Largest admitted original relative opening, in metres.
    pub fn maximum_opening_m(&self) -> f64 {
        self.maximum_opening_m
    }
    /// Positive dimensionless coefficient shared across all admitted aperture regimes.
    pub fn discharge_coefficient(&self) -> f64 {
        self.discharge_coefficient
    }
    /// Explicit positive gravitational acceleration, in m/s².
    pub fn gravity_m_per_s2(&self) -> f64 {
        self.gravity_m_per_s2
    }
    /// Complete immutable coefficient and physical-envelope evidence.
    pub fn applicability(&self) -> &GateApplicability {
        &self.applicability
    }
}
