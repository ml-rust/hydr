//! Outward-rounded F64 arithmetic and adjustment reduction bounds.

mod error;
mod lanes;
mod reduction;

pub(crate) use error::OutwardError;
pub(crate) use reduction::{AdjustmentReduction, adjustment_volume_totals};

use error::{finite, nonnegative};

/// One stored-F64 binade spacing, including the smallest subnormal quantum.
pub(crate) fn rounding_bound(value: f64) -> Result<f64, OutwardError> {
    finite(value, "rounding-bound input")?;
    finite(
        f64::EPSILON * value.abs() + f64::from_bits(1),
        "rounding bound",
    )
}

pub(crate) fn upward_add(a: f64, b: f64) -> Result<f64, OutwardError> {
    nonnegative(a, "outward addition first operand")?;
    nonnegative(b, "outward addition second operand")?;
    let result = finite(a + b, "outward addition intermediate")?;
    finite(result + rounding_bound(result)?, "outward addition")
}

pub(crate) fn upward_mul(a: f64, b: f64) -> Result<f64, OutwardError> {
    nonnegative(a, "outward product first operand")?;
    nonnegative(b, "outward product second operand")?;
    let result = finite(a * b, "outward product intermediate")?;
    finite(result + rounding_bound(result)?, "outward product")
}

/// Zero charges preserve the accepted accumulator exactly, including signed zero.
pub(crate) fn add_adjustment_charge(current: f64, step: f64) -> Result<f64, OutwardError> {
    nonnegative(current, "accepted adjustment charge")?;
    nonnegative(step, "step adjustment charge")?;
    if step == 0.0 {
        Ok(current)
    } else {
        upward_add(current, step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_charge_preserves_bits_and_tiny_charge_advances() -> Result<(), OutwardError> {
        assert_eq!(
            add_adjustment_charge(-0.0, 0.0)?.to_bits(),
            (-0.0f64).to_bits()
        );
        assert_eq!(1.0 + f64::from_bits(1), 1.0);
        assert!(add_adjustment_charge(1.0, f64::from_bits(1))? > 1.0);
        assert!(add_adjustment_charge(f64::MAX, 1.0).is_err());
        Ok(())
    }
}
