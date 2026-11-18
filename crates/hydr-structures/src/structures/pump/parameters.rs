//! Immutable pump curve, suction bound, speed policy, free outlet, and physical applicability.

use super::{PumpApplicability, PumpCurve, PumpFreeOutlet, PumpSpeedPolicy};
use crate::structures::{StructureError, StructureInputErrorKind};

/// Explicit reference-curve proposal model without evolving pump or storage state.
#[derive(Clone, Debug, PartialEq)]
pub struct PumpParameters {
    curve: PumpCurve,
    minimum_suction_stage_m: f64,
    speed_policy: PumpSpeedPolicy,
    applicability: PumpApplicability,
    free_outlet: Option<PumpFreeOutlet>,
}
impl PumpParameters {
    /// Check explicit suction bound, supported speed policy, and complete physical evidence.
    ///
    /// The pump has no free outlet. Add one with [`Self::with_free_outlet`].
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
            free_outlet: None,
        })
    }
    /// Check and attach a free delivery outlet, replacing any earlier one.
    ///
    /// Lift becomes `max(receiving stage, outlet) - donor stage`. A dry receiver whose bed lies at
    /// or below the outlet takes delivery.
    pub fn with_free_outlet(mut self, outlet: PumpFreeOutlet) -> Result<Self, StructureError> {
        outlet.check()?;
        self.free_outlet = Some(outlet);
        Ok(self)
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
    /// Declared free delivery outlet. Without one, a running pump needs a wet receiver.
    pub fn free_outlet(&self) -> Option<&PumpFreeOutlet> {
        self.free_outlet.as_ref()
    }
}
