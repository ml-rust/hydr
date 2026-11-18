//! Conservative cumulative-scale ceiling independent of imported lower-bound claims.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::{SubcatchmentError, SubcatchmentLedger};
use crate::numerics::outward;
use crate::runoff::tensor::subcatchment::TensorSubcatchmentContext;

fn rounding_bound(value: f64) -> Result<f64, SubcatchmentError> {
    outward::rounding_bound(value).map_err(SubcatchmentError::from_outward)
}

fn upward_add(a: f64, b: f64) -> Result<f64, SubcatchmentError> {
    outward::upward_add(a, b).map_err(SubcatchmentError::from_outward)
}

fn cumulative_count<R: Runtime<DType = DType>>(
    context: &TensorSubcatchmentContext<R>,
    steps: usize,
) -> Result<usize, SubcatchmentError> {
    context
        .definitions
        .len()
        .checked_mul(3)
        .and_then(|n| n.checked_add(7))
        .and_then(|n| n.checked_add(steps))
        .ok_or_else(|| SubcatchmentError::input("cumulative accounting count overflows"))
}

/// Lower bound for the exact nonnegative sum behind rounded lifetime counters.
pub(super) fn adjustment_lower<R>(
    context: &TensorSubcatchmentContext<R>,
    value: f64,
    steps: usize,
) -> Result<f64, SubcatchmentError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let additions = cumulative_count(context, steps)?;
    counter_lower(value, additions)
}

fn counter_lower(value: f64, additions: usize) -> Result<f64, SubcatchmentError> {
    if additions as u128 > (1u128 << 53) {
        return Err(SubcatchmentError::input(
            "accounting addition count exceeds exact F64 integer range",
        ));
    }
    let x = additions as f64 * f64::EPSILON;
    let d0 = 1.0 - x;
    let denominator = d0 - rounding_bound(d0)?;
    if !(0.0..1.0).contains(&x) || !denominator.is_finite() || denominator <= 0.0 {
        return Err(SubcatchmentError::input(
            "cumulative accounting denominator is not positive finite",
        ));
    }
    if value == 0.0 {
        return Ok(0.0);
    }
    let product = value * denominator;
    Ok((product - rounding_bound(product)?).max(0.0))
}

/// Every positive traffic stream consumes at least EPSILON/2 of its exact volume.
pub(super) fn transfer_charge_lower<R>(
    context: &TensorSubcatchmentContext<R>,
    ledger: &SubcatchmentLedger,
    steps: usize,
) -> Result<f64, SubcatchmentError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let additions = cumulative_count(context, steps)?;
    let internal_additions = additions
        .checked_add(2)
        .ok_or_else(|| SubcatchmentError::input("internal traffic addition count overflows"))?;
    let external = counter_lower(ledger.external_runon_volume_m3, additions)?;
    let internal = counter_lower(ledger.internal_transfer_volume_m3, internal_additions)?;
    let total = external + internal;
    if !total.is_finite() {
        return Err(SubcatchmentError::numerical(
            "traffic lower-bound sum overflows",
        ));
    }
    let lower = (total - rounding_bound(total)?).max(0.0);
    let product = lower * (f64::EPSILON / 2.0);
    Ok((product - rounding_bound(product)?).max(0.0))
}

fn counter_upper(value: f64, additions: usize) -> Result<f64, SubcatchmentError> {
    if additions as u128 > (1u128 << 53) {
        return Err(SubcatchmentError::input(
            "accounting addition count exceeds exact F64 integer range",
        ));
    }
    let x = additions as f64 * f64::EPSILON;
    let denominator0 = 1.0 - x;
    let denominator = denominator0 - rounding_bound(denominator0)?;
    if !(0.0..1.0).contains(&x) || !denominator.is_finite() || denominator <= 0.0 {
        return Err(SubcatchmentError::input(
            "cumulative accounting denominator is not positive finite",
        ));
    }
    if value == 0.0 {
        return Ok(0.0);
    }
    let quotient = value / denominator;
    let upper = quotient + rounding_bound(quotient)?;
    if !upper.is_finite() {
        return Err(SubcatchmentError::numerical(
            "physical-scale counter ceiling overflows",
        ));
    }
    Ok(upper)
}

pub(super) fn physical_scale_ceiling<R>(
    context: &TensorSubcatchmentContext<R>,
    ledger: &SubcatchmentLedger,
    steps: usize,
) -> Result<f64, SubcatchmentError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let flat = context
        .definitions
        .len()
        .checked_mul(3)
        .ok_or_else(|| SubcatchmentError::input("component support count overflows"))?;
    let initial = flat
        .checked_add(7)
        .ok_or_else(|| SubcatchmentError::input("initial reduction count overflows"))?;
    let cumulative = initial
        .checked_add(steps)
        .ok_or_else(|| SubcatchmentError::input("cumulative reduction count overflows"))?;
    let mut ceiling = counter_upper(ledger.initial_storage_m3, initial)?;
    let internal_count = cumulative
        .checked_add(2)
        .ok_or_else(|| SubcatchmentError::input("internal traffic addition count overflows"))?;
    for (value, count) in [
        (ledger.applied_rain_volume_m3, cumulative),
        (ledger.external_runon_volume_m3, cumulative),
        (ledger.internal_transfer_volume_m3, internal_count),
    ] {
        let upper = counter_upper(value, count)?;
        if upper > 0.0 {
            ceiling = if ceiling == 0.0 {
                upper
            } else {
                upward_add(ceiling, upper)?
            };
        }
    }
    Ok(ceiling)
}

/// Symmetric errors for sums of original represented volume operands.
///
/// Upstream depth-area products and parent scalar arithmetic remain outside these reduction/counter bounds.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct SubcatchmentAccountingBounds {
    pub storage_before_error_m3: f64,
    pub storage_after_error_m3: f64,
    pub rain_error_m3: f64,
    pub external_runon_error_m3: f64,
    pub infiltration_error_m3: f64,
    pub evaporation_error_m3: f64,
    pub outlet_error_m3: f64,
    pub soil_adjustment_error_m3: f64,
    pub combined_error_m3: f64,
}

/// Retain both directions of the existing nonnegative exact-total envelope.
fn counter_error(value: f64, additions: usize) -> Result<f64, SubcatchmentError> {
    if !value.is_finite() || value < 0.0 {
        return Err(SubcatchmentError::numerical(
            "accounting envelope requires a finite nonnegative magnitude",
        ));
    }
    let lower = counter_lower(value, additions)?;
    let upper = counter_upper(value, additions)?;
    if value == 0.0 {
        return Ok(0.0);
    }
    let below = outward_difference(value, lower)?;
    let above = outward_difference(upper, value)?;
    Ok(below.max(above))
}

fn outward_difference(greater: f64, lesser: f64) -> Result<f64, SubcatchmentError> {
    let difference = greater - lesser;
    if !difference.is_finite() || difference < 0.0 {
        return Err(SubcatchmentError::numerical(
            "accounting envelope endpoints have an invalid finite order",
        ));
    }
    let upper = difference + rounding_bound(difference)?;
    if !upper.is_finite() {
        return Err(SubcatchmentError::numerical(
            "accounting envelope error overflows. Rescale physical support",
        ));
    }
    Ok(upper)
}

fn combined_bounds(values: [f64; 8]) -> Result<SubcatchmentAccountingBounds, SubcatchmentError> {
    let [
        before,
        after,
        rain,
        external,
        infiltration,
        evaporation,
        outlet,
        adjustment,
    ] = values;
    let mut bounds = SubcatchmentAccountingBounds {
        storage_before_error_m3: before,
        storage_after_error_m3: after,
        rain_error_m3: rain,
        external_runon_error_m3: external,
        infiltration_error_m3: infiltration,
        evaporation_error_m3: evaporation,
        outlet_error_m3: outlet,
        soil_adjustment_error_m3: adjustment,
        combined_error_m3: 0.0,
    };
    for value in [
        bounds.storage_before_error_m3,
        bounds.storage_after_error_m3,
        bounds.rain_error_m3,
        bounds.external_runon_error_m3,
        bounds.infiltration_error_m3,
        bounds.evaporation_error_m3,
        bounds.outlet_error_m3,
        bounds.soil_adjustment_error_m3,
    ] {
        if value > 0.0 {
            bounds.combined_error_m3 = if bounds.combined_error_m3 == 0.0 {
                value
            } else {
                upward_add(bounds.combined_error_m3, value)?
            };
        }
    }
    Ok(bounds)
}

pub(super) fn interval_bounds<R: Runtime<DType = DType>>(
    context: &TensorSubcatchmentContext<R>,
    totals: &super::accounting::StepTotals,
    adjustment_absolute_charge: f64,
) -> Result<SubcatchmentAccountingBounds, SubcatchmentError> {
    let count = cumulative_count(context, 0)?;
    // Signed reduction error depends on absolute operands, never their cancellation-prone signed total.
    combined_bounds([
        counter_error(totals.before, count)?,
        counter_error(totals.after, count)?,
        counter_error(totals.rain, count)?,
        counter_error(totals.external, count)?,
        counter_error(totals.infiltration, count)?,
        counter_error(totals.evaporation, count)?,
        counter_error(totals.outlet, count)?,
        counter_error(adjustment_absolute_charge, count)?,
    ])
}

pub(super) fn lifetime_bounds<R: Runtime<DType = DType>>(
    context: &TensorSubcatchmentContext<R>,
    ledger: &SubcatchmentLedger,
    steps: usize,
) -> Result<SubcatchmentAccountingBounds, SubcatchmentError> {
    let reduction = cumulative_count(context, 0)?;
    let cumulative = cumulative_count(context, steps)?;
    lifetime_counter_bounds(ledger, reduction, cumulative)
}

fn lifetime_counter_bounds(
    ledger: &SubcatchmentLedger,
    reduction: usize,
    cumulative: usize,
) -> Result<SubcatchmentAccountingBounds, SubcatchmentError> {
    // The accumulated absolute correction charge bounds signed reduction and counter operands.
    combined_bounds([
        counter_error(ledger.initial_storage_m3, reduction)?,
        counter_error(ledger.storage_m3, reduction)?,
        counter_error(ledger.applied_rain_volume_m3, cumulative)?,
        counter_error(ledger.external_runon_volume_m3, cumulative)?,
        counter_error(ledger.infiltration_volume_m3, cumulative)?,
        counter_error(ledger.evaporation_volume_m3, cumulative)?,
        counter_error(ledger.outlet_volume_m3, cumulative)?,
        counter_error(ledger.soil_water_adjustment_abs_bound_m3, cumulative)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runoff::{SubcatchmentError, SubcatchmentLedger};

    #[test]
    fn reported_rounding_in_both_directions_keeps_exact_sum_enclosed()
    -> Result<(), SubcatchmentError> {
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
    fn subnormal_transition_encloses_exact_minimum_quantum_errors() -> Result<(), SubcatchmentError>
    {
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
}
