//! Conservative cumulative-scale ceiling independent of imported lower-bound claims.

use super::{SubcatchmentError, SubcatchmentLedger};
use crate::cpu::subcatchment::CpuSubcatchmentContext;
use crate::soil::admission::{rounding_bound, upward_add};

fn cumulative_count(
    context: &CpuSubcatchmentContext,
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
pub(super) fn adjustment_lower(
    context: &CpuSubcatchmentContext,
    value: f64,
    steps: usize,
) -> Result<f64, SubcatchmentError> {
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
pub(super) fn transfer_charge_lower(
    context: &CpuSubcatchmentContext,
    ledger: &SubcatchmentLedger,
    steps: usize,
) -> Result<f64, SubcatchmentError> {
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

pub(super) fn physical_scale_ceiling(
    context: &CpuSubcatchmentContext,
    ledger: &SubcatchmentLedger,
    steps: usize,
) -> Result<f64, SubcatchmentError> {
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
