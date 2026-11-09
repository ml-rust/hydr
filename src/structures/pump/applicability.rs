//! Explicit stage-head, affinity, isolation, and operating-envelope assumptions.

use crate::structures::{StructureError, StructureInputErrorKind};

/// Physical applicability evidence without an efficiency, cavitation, or control model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PumpApplicability {
    /// Explicit quiescent common-datum endpoint-stage interpretation.
    pub endpoint_stage_head_assertion: String,
    /// Included connection losses, velocity-head differences, and curve reference locations.
    pub connection_losses_and_velocity_head_account: String,
    /// Isolation evidence excluding forward siphoning and reverse flow while stopped.
    pub stopped_isolation_assertion: String,
    /// Source operating envelope, including suction/receiver assumptions.
    pub operating_envelope: String,
}
impl PumpApplicability {
    pub(crate) fn check(&self) -> Result<(), StructureError> {
        for (value, field) in [
            (
                &self.endpoint_stage_head_assertion,
                "endpoint_stage_head_assertion",
            ),
            (
                &self.connection_losses_and_velocity_head_account,
                "connection_losses_and_velocity_head_account",
            ),
            (
                &self.stopped_isolation_assertion,
                "stopped_isolation_assertion",
            ),
            (&self.operating_envelope, "operating_envelope"),
        ] {
            crate::structures::definition::text(value, field)?;
        }
        Ok(())
    }
}

/// Reference-only operation or an explicitly supplied affinity model.
#[derive(Clone, Debug, PartialEq)]
pub enum PumpSpeedPolicy {
    /// Exact reference ratio one only, without an implied affinity assumption.
    ReferenceOnly,
    /// Declared affinity scaling over a finite positive interval containing reference ratio one.
    DeclaredAffinity {
        /// Inclusive positive supported ratio.
        minimum_ratio: f64,
        /// Inclusive positive supported ratio.
        maximum_ratio: f64,
        /// Explicit source-specific affinity assumptions.
        applicability: String,
    },
}
impl PumpSpeedPolicy {
    pub(crate) fn check(&self) -> Result<(), StructureError> {
        if let Self::DeclaredAffinity {
            minimum_ratio,
            maximum_ratio,
            applicability,
        } = self
        {
            crate::structures::definition::text(applicability, "affinity applicability")?;
            if !minimum_ratio.is_finite()
                || *minimum_ratio <= 0.0
                || !maximum_ratio.is_finite()
                || *maximum_ratio <= 0.0
                || minimum_ratio > maximum_ratio
                || *minimum_ratio > 1.0
                || *maximum_ratio < 1.0
            {
                return Err(StructureError::input(
                    StructureInputErrorKind::Definition,
                    format!(
                        "speed policy={self:?}: require positive finite ordered ratios containing reference ratio one"
                    ),
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn range(&self) -> (f64, f64) {
        match self {
            Self::ReferenceOnly => (1.0, 1.0),
            Self::DeclaredAffinity {
                minimum_ratio,
                maximum_ratio,
                ..
            } => (*minimum_ratio, *maximum_ratio),
        }
    }
}
