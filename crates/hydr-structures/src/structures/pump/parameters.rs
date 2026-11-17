//! Immutable pump curve, suction bound, speed policy, and physical applicability.

use super::{PumpApplicability, PumpCurve, PumpSpeedPolicy};
use crate::structures::{StructureError, StructureInputErrorKind};

/// Explicit reference-curve proposal model without evolving pump or storage state.
#[derive(Clone, Debug, PartialEq)]
pub struct PumpParameters {
    curve: PumpCurve,
    minimum_suction_stage_m: f64,
    speed_policy: PumpSpeedPolicy,
    applicability: PumpApplicability,
}
impl PumpParameters {
    /// Check explicit suction bound, supported speed policy, and complete physical evidence.
    pub fn new(
        curve: PumpCurve,
        minimum_suction_stage_m: f64,
        speed_policy: PumpSpeedPolicy,
        applicability: PumpApplicability,
    ) -> Result<Self, StructureError> {
        if !minimum_suction_stage_m.is_finite() {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!(
                    "minimum_suction_stage_m={minimum_suction_stage_m}: supply a finite common-datum suction bound"
                ),
            ));
        }
        speed_policy.check()?;
        applicability.check()?;
        Ok(Self {
            curve,
            minimum_suction_stage_m,
            speed_policy,
            applicability,
        })
    }
    /// Complete closed reference-speed curve.
    pub fn curve(&self) -> &PumpCurve {
        &self.curve
    }
    /// Running donor stage must lie strictly above this common-datum bound.
    pub fn minimum_suction_stage_m(&self) -> f64 {
        self.minimum_suction_stage_m
    }
    /// Immutable reference-only or explicitly declared affinity policy.
    pub fn speed_policy(&self) -> &PumpSpeedPolicy {
        &self.speed_policy
    }
    /// Complete physical stage-head, loss, isolation, and operating-envelope evidence.
    pub fn applicability(&self) -> &PumpApplicability {
        &self.applicability
    }
}
