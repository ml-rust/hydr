//! Original immutable trajectory inputs and static admission.

use super::accuracy::{GateMotionLimits, GateTrajectoryAccuracy};
use crate::structures::{StructureError, StructureInputErrorKind};

/// One original SI gate-opening coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateOpeningKnot {
    /// Nonnegative elapsed time in seconds.
    pub time_s: f64,
    /// Nonnegative opening in metres.
    pub opening_m: f64,
}

/// Immutable continuous piecewise-linear opening with explicit motion and accuracy policies.
#[derive(Clone, Debug, PartialEq)]
pub struct GateOpeningTrajectory {
    knots: Vec<GateOpeningKnot>,
    motion_limits: GateMotionLimits,
    accuracy: GateTrajectoryAccuracy,
    provenance: String,
}
impl GateOpeningTrajectory {
    /// Check original inputs. Binding checks motion arithmetic and the actual gate range.
    pub fn new(
        knots: Vec<GateOpeningKnot>,
        motion_limits: GateMotionLimits,
        accuracy: GateTrajectoryAccuracy,
        provenance: String,
    ) -> Result<Self, StructureError> {
        crate::structures::definition::text(&provenance, "gate trajectory provenance")?;
        motion_limits.check()?;
        accuracy.check()?;
        if knots.len() < 2 {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                "gate trajectory needs at least two knots: supply explicit closed support",
            ));
        }
        for (index, knot) in knots.iter().enumerate() {
            if !knot.time_s.is_finite()
                || knot.time_s < 0.0
                || !knot.opening_m.is_finite()
                || knot.opening_m < 0.0
            {
                return Err(StructureError::input(
                    StructureInputErrorKind::Definition,
                    format!(
                        "gate knot={index},value={knot:?}: supply finite nonnegative time and opening"
                    ),
                ));
            }
        }
        for (index, pair) in knots.windows(2).enumerate() {
            if pair[1].time_s <= pair[0].time_s {
                return Err(StructureError::input(
                    StructureInputErrorKind::Definition,
                    format!(
                        "gate segment={index},times={:?}/{:?}: supply strictly increasing original times",
                        pair[0].time_s, pair[1].time_s
                    ),
                ));
            }
        }
        Ok(Self {
            knots,
            motion_limits,
            accuracy,
            provenance,
        })
    }
    /// Complete original coordinates without extrapolation.
    pub fn knots(&self) -> &[GateOpeningKnot] {
        &self.knots
    }
    /// Original directional speed limits.
    pub fn motion_limits(&self) -> GateMotionLimits {
        self.motion_limits
    }
    /// Original numerical opening-error policy.
    pub fn accuracy(&self) -> GateTrajectoryAccuracy {
        self.accuracy
    }
    /// Original source and operating-envelope evidence.
    pub fn provenance(&self) -> &str {
        &self.provenance
    }
}
