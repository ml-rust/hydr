//! Checked blocked share of a declared unblocked opening.

use crate::structures::StructureError;

/// Blocked share `f` of a declared unblocked opening, in `[0, 1)`.
///
/// A fully blocked structure is [`StructureCondition::Closed`](super::StructureCondition::Closed),
/// never a fraction of one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockageFraction(f64);

impl BlockageFraction {
    /// No blockage. Every effective value keeps the bits of its base value.
    pub const NONE: Self = Self(0.0);

    /// Checks a finite blocked share with `0 <= f < 1`. `-0.0` becomes `+0.0`.
    pub fn new(blocked_share: f64) -> Result<Self, StructureError> {
        if !blocked_share.is_finite() || !(0.0..1.0).contains(&blocked_share) {
            return Err(StructureError::definition(format!(
                "blocked_share={blocked_share}: supply a finite blocked share with 0 <= f < 1. Declare StructureCondition::Closed for a fully blocked structure"
            )));
        }
        Ok(Self(blocked_share + 0.0))
    }

    /// Blocked share `f`.
    pub fn blocked_share(self) -> f64 {
        self.0
    }

    /// Open share `fl(1 - f)`, in `(0, 1]`.
    pub fn open_share(self) -> f64 {
        1.0 - self.0
    }

    /// Effective value `fl(x · fl(1 - f))` of a base value `x`.
    pub(crate) fn apply(self, base: f64) -> f64 {
        base * self.open_share()
    }
}

#[cfg(test)]
mod tests {
    use super::BlockageFraction;

    #[test]
    fn negative_zero_becomes_positive_zero() -> Result<(), Box<dyn std::error::Error>> {
        let fraction = BlockageFraction::new(-0.0)?;
        assert_eq!(fraction.blocked_share().to_bits(), 0.0_f64.to_bits());
        assert_eq!(fraction, BlockageFraction::NONE);
        Ok(())
    }

    #[test]
    fn zero_fraction_keeps_base_bits() -> Result<(), Box<dyn std::error::Error>> {
        let none = BlockageFraction::new(0.0)?;
        for base in [f64::MIN_POSITIVE, 1e-300, 0.1, 1.0, 3.7, 1e300, f64::MAX] {
            assert_eq!(none.apply(base).to_bits(), base.to_bits());
        }
        Ok(())
    }

    #[test]
    fn domain_outside_half_open_unit_rejects() {
        for value in [1.0, 1.5, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = BlockageFraction::new(value).err();
            let message = error.map(|error| error.to_string()).unwrap_or_default();
            assert!(
                message.contains("blocked_share=") && message.contains("Closed"),
                "value={value}: {message}"
            );
        }
    }

    #[test]
    fn largest_share_below_one_admits() -> Result<(), Box<dyn std::error::Error>> {
        let fraction = BlockageFraction::new(1.0 - f64::EPSILON / 2.0)?;
        assert!(fraction.open_share() > 0.0);
        Ok(())
    }
}
