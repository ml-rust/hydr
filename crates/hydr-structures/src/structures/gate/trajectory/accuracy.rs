//! Explicit directional limits and numerical opening budgets.

use crate::structures::{StructureError, StructureInputErrorKind};

/// Finite nonnegative directional speed limits. Zero prohibits motion in that direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateMotionLimits {
    /// Maximum increasing-opening speed in metres per second.
    pub maximum_opening_speed_m_per_s: f64,
    /// Maximum decreasing-opening speed in metres per second.
    pub maximum_closing_speed_m_per_s: f64,
}
impl GateMotionLimits {
    pub(super) fn check(self) -> Result<(), StructureError> {
        check(
            self.maximum_opening_speed_m_per_s,
            "maximum_opening_speed_m_per_s",
        )?;
        check(
            self.maximum_closing_speed_m_per_s,
            "maximum_closing_speed_m_per_s",
        )
    }
}
/// Explicit absolute-or-relative receipt admission. Neither field has an implicit minimum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GateTrajectoryAccuracy {
    /// Maximum absolute opening-error receipt in metres.
    pub maximum_absolute_opening_error_m: f64,
    /// Maximum outward relative opening-error receipt.
    pub maximum_relative_opening_error: f64,
}
impl GateTrajectoryAccuracy {
    pub(super) fn check(self) -> Result<(), StructureError> {
        check(
            self.maximum_absolute_opening_error_m,
            "maximum_absolute_opening_error_m",
        )?;
        check(
            self.maximum_relative_opening_error,
            "maximum_relative_opening_error",
        )
    }
}
fn check(value: f64, name: &str) -> Result<(), StructureError> {
    if !value.is_finite() || value < 0.0 {
        return Err(StructureError::input(
            StructureInputErrorKind::Definition,
            format!("{name}={value}: supply an explicit finite nonnegative limit"),
        ));
    }
    Ok(())
}
