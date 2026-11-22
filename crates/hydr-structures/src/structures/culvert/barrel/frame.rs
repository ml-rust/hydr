//! Exact power-of-two frame of the arc admission rules.
//!
//! The frame scales the rise into `[1, 2)`. Every tangency rule is homogeneous in length, so
//! it holds in the frame exactly when it holds in metres. No square or product of the rules
//! underflows or overflows there.

use super::enclosure::Enclosure;
use crate::structures::StructureError;

/// Exact power-of-two scaling that takes a rise into `[1, 2)`.
pub(super) struct Frame {
    scale: f64,
    unscale: f64,
}

impl Frame {
    /// The frame of `rise_m`, or a rejection when a dimension of `dimensions` does not scale
    /// exactly into it.
    pub(super) fn new(
        rise_m: f64,
        dimensions: &[f64],
        fields: &str,
    ) -> Result<Self, StructureError> {
        // An admitted rise is a normal F64, so its biased exponent gives the scale exactly.
        let biased = i32::try_from((rise_m.to_bits() >> 52) & 0x7ff).unwrap_or(1023);
        let exponent = biased - 1023;
        let frame = Self {
            scale: 2f64.powi(-exponent),
            unscale: 2f64.powi(exponent),
        };
        if dimensions
            .iter()
            .any(|&value| frame.back(frame.scaled(value)) != value)
        {
            return Err(StructureError::definition(format!(
                "{fields}: the dimensions differ in size by more than F64 can scale to a unit rise. Supply dimensions closer in size"
            )));
        }
        Ok(frame)
    }

    /// `value` in the frame.
    pub(super) fn scaled(&self, value: f64) -> f64 {
        value * self.scale
    }

    /// A frame length `value` back in metres, exact unless it overflows.
    pub(super) fn back(&self, value: f64) -> f64 {
        value * self.unscale
    }

    /// An enclosure in the frame back in metres.
    pub(super) fn back_enclosure(&self, value: Enclosure) -> Enclosure {
        Enclosure {
            lower: self.back(value.lower),
            upper: self.back(value.upper),
        }
    }
}
