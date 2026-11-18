use super::AdjustmentReduction;
use crate::numerics::outward::error::{OutwardError, finite, nonnegative};
use crate::numerics::outward::{rounding_bound, upward_add, upward_mul};

impl AdjustmentReduction {
    /// Bound absolute error in a reported nonnegative, unit-area F64 sum.
    ///
    /// Assume IEEE round-to-nearest addition with gradual underflow and at most N-1 additions.
    /// Unit-area multiplication is exact. This bound does not cover rounded area products.
    #[inline]
    pub fn unit_sum_error_bound(self, reported: f64) -> Result<f64, OutwardError> {
        nonnegative(reported, "reported unit-area sum")?;
        if self.count == 0 && reported != 0.0 {
            return Err(OutwardError::input(
                "empty unit-area reduction reports a nonzero sum",
            ));
        }
        if self.count <= 1 || reported == 0.0 {
            return Ok(0.0);
        }
        // With k <= N-1, u=EPSILON/2, q=smallest subnormal, and exact sum S:
        // |reported-S| <= (k*u*S + k*q/2)/(1-k*u).
        // Substitution S <= reported+|reported-S| gives denominator 1-2*k*u.
        // The cached denominator is <= 1-(N+7)*EPSILON, hence <= 1-2*k*u.
        // Numerator (N+7)*EPSILON*reported + N*q dominates both error directions.
        // This is the same conservative count envelope used by storage accounting.
        let additions = self
            .count
            .checked_add(7)
            .ok_or_else(|| OutwardError::input("unit-area reduction count overflows usize"))?;
        let relative = upward_mul(reported, additions as f64 * f64::EPSILON)?;
        let absolute = upward_mul(self.count as f64, f64::from_bits(1))?;
        let numerator = upward_add(relative, absolute)?;
        let quotient = finite(
            numerator / self.denominator,
            "unit-area reduction error quotient",
        )?;
        finite(
            quotient + rounding_bound(quotient)?,
            "unit-area absolute reduction error bound",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    use numr::tensor::Tensor;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone())
            .with_parallelism(ParallelismConfig::new(Some(1), Some(32)));
        (device, client)
    }

    fn integer_units(value: f64, unit_exponent: i32) -> Result<u128, OutwardError> {
        if !value.is_finite() || value < 0.0 {
            return Err(OutwardError::numerical(
                "dyadic oracle requires finite nonnegative values",
            ));
        }
        if value == 0.0 {
            return Ok(0);
        }
        let bits = value.to_bits();
        let encoded_exponent = ((bits >> 52) & 0x7ff) as i32;
        let fraction = bits & ((1u64 << 52) - 1);
        let (significand, exponent) = if encoded_exponent == 0 {
            (fraction as u128, -1074)
        } else {
            (
                ((1u64 << 52) | fraction) as u128,
                encoded_exponent - 1023 - 52,
            )
        };
        let shift = exponent - unit_exponent;
        if shift >= 0 {
            let multiplier = 1u128
                .checked_shl(shift as u32)
                .ok_or_else(|| OutwardError::numerical("dyadic oracle shift exceeds u128"))?;
            significand
                .checked_mul(multiplier)
                .ok_or_else(|| OutwardError::numerical("dyadic oracle integer exceeds u128"))
        } else {
            let bits = (-shift) as u32;
            let divisor = 1u128
                .checked_shl(bits)
                .ok_or_else(|| OutwardError::numerical("dyadic oracle divisor exceeds u128"))?;
            if significand % divisor != 0 {
                return Err(OutwardError::numerical(
                    "reported sum is not an integer multiple of the oracle unit",
                ));
            }
            Ok(significand / divisor)
        }
    }

    fn check_case(
        units: &[u128],
        unit_exponent: i32,
        strided: bool,
    ) -> Result<(u128, u128), OutwardError> {
        let (device, client) = setup();
        let unit = if unit_exponent == -1074 {
            f64::from_bits(1)
        } else {
            2.0f64.powi(unit_exponent)
        };
        let exact = units.iter().try_fold(0u128, |sum, value| {
            sum.checked_add(*value)
                .ok_or_else(|| OutwardError::numerical("exact dyadic sum exceeds u128"))
        })?;
        let values: Vec<_> = units.iter().map(|value| *value as f64 * unit).collect();
        for (value, expected) in values.iter().zip(units) {
            assert_eq!(integer_units(*value, unit_exponent)?, *expected);
        }
        let tensor = if strided {
            let padded: Vec<_> = values.iter().flat_map(|value| [99.0, *value]).collect();
            Tensor::<CpuRuntime>::from_slice(&padded, &[units.len(), 2], &device)?
                .narrow(1, 1, 1)?
                .squeeze(Some(1))
        } else {
            Tensor::<CpuRuntime>::from_slice(&values, &[values.len()], &device)?
        };
        let reduction = AdjustmentReduction::new(units.len())?;
        let reported = reduction.totals(&client, &tensor, 1.0)?.0;
        let reported_units = integer_units(reported, unit_exponent)?;
        let difference = reported_units.abs_diff(exact);
        assert!(difference <= (1u128 << 53));
        let exact_error = difference as f64 * unit;
        let bound = reduction.unit_sum_error_bound(reported)?;
        assert!(bound.is_finite());
        assert!(
            bound >= exact_error,
            "N={},strided={strided},reported_units={reported_units},exact_units={exact},error={exact_error:e},bound={bound:e}",
            units.len()
        );
        Ok((reported_units, exact))
    }

    #[test]
    fn symmetric_bound_covers_both_directions_of_two_value_rounding() -> Result<(), OutwardError> {
        for strided in [false, true] {
            let (reported, exact) = check_case(&[1u128 << 54, 3], -54, strided)?;
            assert!(reported > exact);
            let (reported, exact) = check_case(&[1u128 << 54, 1], -54, strided)?;
            assert!(reported < exact);
        }
        Ok(())
    }

    #[test]
    fn long_and_strided_reductions_enclose_exact_dyadic_sums() -> Result<(), OutwardError> {
        for count in [31, 32, 33, 127, 128, 129, 257, 513] {
            for seed in 0..5 {
                let mut units: Vec<_> = (0..count)
                    .map(|index| match (index + seed) % 5 {
                        0 => 1u128 << 54,
                        1 => 3,
                        2 => 1,
                        3 => 1u128 << 30,
                        _ => 5,
                    })
                    .collect();
                for strided in [false, true] {
                    check_case(&units, -54, strided)?;
                }
                units.reverse();
                check_case(&units, -54, false)?;
            }
        }
        Ok(())
    }

    #[test]
    fn subnormal_and_normal_transition_sums_keep_an_absolute_allowance() -> Result<(), OutwardError>
    {
        for count in [2, 31, 32, 33, 257] {
            for strided in [false, true] {
                check_case(&vec![1; count], -1074, strided)?;
                let units: Vec<_> = (0..count)
                    .map(|index| (1u128 << 48) + (index % 7) as u128)
                    .collect();
                check_case(&units, -1074, strided)?;
            }
        }
        Ok(())
    }

    #[test]
    fn exact_zero_and_single_values_require_no_reduction_error_charge() -> Result<(), OutwardError>
    {
        assert_eq!(AdjustmentReduction::new(0)?.unit_sum_error_bound(0.0)?, 0.0);
        assert!(
            AdjustmentReduction::new(0)?
                .unit_sum_error_bound(1.0)
                .is_err()
        );
        for value in [0.0, -0.0, f64::from_bits(1), 1.0, f64::MAX] {
            assert_eq!(
                AdjustmentReduction::new(1)?.unit_sum_error_bound(value)?,
                0.0
            );
        }
        assert_eq!(
            AdjustmentReduction::new(257)?.unit_sum_error_bound(0.0)?,
            0.0
        );
        for value in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(
                AdjustmentReduction::new(2)?
                    .unit_sum_error_bound(value)
                    .is_err()
            );
        }
        Ok(())
    }
}
