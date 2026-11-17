//! Explicit closed-pool definitions, transfers, and dimensional controls.

use super::{LevelPoolCurve, LevelPoolError};

/// One connected approximately level physical storage claim.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolElement {
    /// Unique nonempty element identity.
    pub id: String,
    /// Complete checked storage curve and reference metadata.
    pub curve: LevelPoolCurve,
    /// Explicit caller evidence for connected approximately level water.
    pub connected_pool_assertion: String,
    /// Unique physical storage-support identity. No geometric nonoverlap proof follows.
    pub storage_support_identity: String,
}
impl LevelPoolElement {
    #[doc(hidden)]
    pub fn same_bits(&self, other: &Self) -> bool {
        self.id == other.id
            && self.connected_pool_assertion == other.connected_pool_assertion
            && self.storage_support_identity == other.storage_support_identity
            && self.curve.same_bits(&other.curve)
    }
}

/// Explicit authoritative accepted storage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelPoolState {
    /// Finite storage volume within this element's closed curve range, in m³.
    pub volume_m3: f64,
}

/// Authoritative interval exchanges without an assumed within-interval arrival order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelPoolInput {
    /// Finite nonnegative incoming volume in m³.
    pub incoming_volume_m3: f64,
    /// Finite nonnegative outgoing volume in m³, bounded by initial donor storage.
    pub outgoing_volume_m3: f64,
}

/// Independent dimensional inverse/physical accounting gates.
#[derive(Clone, Debug, PartialEq)]
pub struct LevelPoolConfig {
    /// Absolute volume tolerance in m³, without a unit-volume floor.
    pub volume_absolute: f64,
    /// Dimensionless relative tolerance against actual storage and transfer scale.
    pub volume_relative: f64,
}
impl Default for LevelPoolConfig {
    fn default() -> Self {
        Self {
            volume_absolute: 1e-9,
            volume_relative: 1e-12,
        }
    }
}
impl LevelPoolConfig {
    #[doc(hidden)]
    pub fn check(&self) -> Result<(), LevelPoolError> {
        if !self.volume_absolute.is_finite()
            || self.volume_absolute < 0.0
            || !self.volume_relative.is_finite()
            || self.volume_relative < 0.0
            || (self.volume_absolute == 0.0 && self.volume_relative == 0.0)
        {
            return Err(LevelPoolError::input(format!(
                "volume_absolute={},volume_relative={}: require finite nonnegative tolerances with positive combined allowance",
                self.volume_absolute, self.volume_relative
            )));
        }
        Ok(())
    }
    #[doc(hidden)]
    pub fn same_bits(&self, other: &Self) -> bool {
        self.volume_absolute.to_bits() == other.volume_absolute.to_bits()
            && self.volume_relative.to_bits() == other.volume_relative.to_bits()
    }
}

pub(crate) fn text(value: &str, field: &str) -> Result<(), LevelPoolError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(LevelPoolError::input(format!(
            "{field}={value:?}: supply nonempty explicit evidence without control characters"
        )));
    }
    Ok(())
}
