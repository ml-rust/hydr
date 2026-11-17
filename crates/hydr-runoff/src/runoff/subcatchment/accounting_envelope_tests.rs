//! Exact dyadic checks for represented-volume reduction and counter envelopes.

use super::{counter_error, counter_lower, counter_upper, lifetime_counter_bounds};
use crate::runoff::{SubcatchmentError, SubcatchmentLedger};

#[test]
fn reported_rounding_in_both_directions_keeps_exact_sum_enclosed() -> Result<(), SubcatchmentError>
{
    let unit = 2.0f64.powi(-54);
    for original_units in [1i128, 3] {
        let reported =
            std::hint::black_box(1.0) + std::hint::black_box(original_units as f64 * unit);
        let reported_units = ((reported - 1.0) / unit) as i128;
        assert_eq!(reported_units, if original_units == 1 { 0 } else { 4 });
        let lower_units = ((counter_lower(reported, 9)? - 1.0) / unit) as i128;
        let upper_units = ((counter_upper(reported, 9)? - 1.0) / unit) as i128;
        assert!(lower_units <= original_units && original_units <= upper_units);
        let exact_error = reported_units.abs_diff(original_units) as f64 * unit;
        assert!(counter_error(reported, 9)? >= exact_error);
    }
    Ok(())
}

#[test]
fn signed_cancellation_uses_absolute_correction_charge() -> Result<(), SubcatchmentError> {
    let unit = 2.0f64.powi(-54);
    let positive = std::hint::black_box(1.0) + std::hint::black_box(unit);
    let reported = positive - std::hint::black_box(1.0);
    assert_eq!(reported, 0.0);
    // Original signed operands sum to unit, while their absolute sum is less than three.
    let mut ledger = SubcatchmentLedger::new(0.0, 0.0);
    ledger.soil_water_adjustment_volume_m3 = reported;
    ledger.soil_water_adjustment_abs_bound_m3 = 3.0;
    let bounds = lifetime_counter_bounds(&ledger, 9, 10)?;
    assert!(bounds.soil_adjustment_error_m3 >= unit);
    assert!(bounds.combined_error_m3 >= bounds.soil_adjustment_error_m3);
    assert_eq!(counter_error(reported, 10)?, 0.0);
    Ok(())
}

fn transition_units(value: f64) -> u128 {
    assert!(value.is_finite() && value >= 0.0);
    let bits = value.to_bits();
    let exponent = bits >> 52;
    let fraction = bits & ((1u64 << 52) - 1);
    assert!(exponent <= 3);
    if exponent == 0 {
        fraction as u128
    } else {
        (((1u64 << 52) | fraction) as u128) << (exponent - 1)
    }
}

#[test]
fn subnormal_transition_encloses_exact_minimum_quantum_errors() -> Result<(), SubcatchmentError> {
    let base = f64::from_bits(0x0020000000000000);
    for small in [1u64, 3] {
        let reported = std::hint::black_box(base) + std::hint::black_box(f64::from_bits(small));
        let exact_units = (1u128 << 53) + u128::from(small);
        let reported_units = transition_units(reported);
        assert_eq!(reported_units.abs_diff(exact_units), 1);
        assert!(transition_units(counter_lower(reported, 9)?) <= exact_units);
        assert!(exact_units <= transition_units(counter_upper(reported, 9)?));
        assert!(transition_units(counter_error(reported, 9)?) >= 1);
    }
    assert_eq!(counter_error(0.0, 9)?, 0.0);
    Ok(())
}

#[cfg(target_pointer_width = "64")]
#[test]
fn excessive_counts_and_nonpositive_denominators_reject() {
    assert!(counter_error(1.0, (1usize << 53) + 1).is_err());
    assert!(counter_error(1.0, 1usize << 52).is_err());
    assert!(counter_error(0.0, usize::MAX).is_err());
}
