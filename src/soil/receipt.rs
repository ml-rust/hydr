//! Shared native/SI source receipts and algebraic import admission.

use super::SoilError;
use crate::numerics::outward::{OutwardError, rounding_bound, upward_add, upward_mul};

/// Native soil admission evidence for one accepted source interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SoilSourceReceipt {
    /// Original returned infiltration rate in ft/s.
    pub native_rate_ft_per_s: f64,
    /// Raw native available supply in feet.
    pub native_available_depth_ft: f64,
    /// Returned native rate times duration, in feet.
    pub native_infiltration_depth_ft: f64,
    /// Positive native availability discrepancy, in feet.
    pub native_excess_depth_ft: f64,
    /// Outward native branch rounding envelope, in feet.
    pub native_roundoff_bound_ft: f64,
    /// Outward SI conversion envelope, in metres.
    pub bridge_roundoff_bound_m: f64,
    /// Combined outward native/SI envelope, in metres.
    pub total_roundoff_bound_m: f64,
}

const METRES_PER_FOOT: f64 = 0.3048;

/// Actual SI intermediates supplied to one native soil call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SoilReceiptInput {
    pub dt_s: f64,
    pub incoming_rate_m_per_s: f64,
    pub ponded_depth_m: f64,
    pub incoming_depth_m: f64,
    pub available_depth_m: f64,
    pub infiltration_depth_m: f64,
    pub adjustment_depth_m: f64,
}

/// Check receipt algebra without replaying antecedent native state.
pub(crate) fn check_source_receipt(
    input: &SoilReceiptInput,
    soil: &SoilSourceReceipt,
) -> Result<(), SoilError> {
    let dt = input.dt_s;
    let incoming_rate = input.incoming_rate_m_per_s;
    let available = input.available_depth_m;
    require(
        dt.is_finite() && dt > 0.0,
        "soil receipt duration",
        format!("dt={dt}: require a positive finite duration"),
    )?;
    nonnegative(
        &[
            incoming_rate,
            input.ponded_depth_m,
            input.incoming_depth_m,
            available,
            input.infiltration_depth_m,
            input.adjustment_depth_m,
        ],
        "soil receipt SI inputs",
    )?;
    require(
        input.incoming_depth_m == incoming_rate * dt
            && available == input.ponded_depth_m + input.incoming_depth_m,
        "soil receipt SI supply",
        "retain the original incoming depth and available supply",
    )?;
    nonnegative(
        &[
            soil.native_rate_ft_per_s,
            soil.native_available_depth_ft,
            soil.native_infiltration_depth_ft,
            soil.native_excess_depth_ft,
            soil.native_roundoff_bound_ft,
            soil.bridge_roundoff_bound_m,
            soil.total_roundoff_bound_m,
        ],
        "soil source receipt",
    )?;
    let a = incoming_rate / METRES_PER_FOOT;
    let dh = input.ponded_depth_m / METRES_PER_FOOT;
    let vh = dh / dt;
    let ia = a + vh;
    let native_available = ia * dt;
    let native_depth = soil.native_rate_ft_per_s * dt;
    let si_rate = soil.native_rate_ft_per_s * METRES_PER_FOOT;
    let si_depth = si_rate * dt;
    nonnegative(
        &[
            a,
            dh,
            vh,
            ia,
            native_available,
            native_depth,
            si_rate,
            si_depth,
        ],
        "soil source bridge",
    )?;
    require(
        soil.native_available_depth_ft == native_available
            && soil.native_infiltration_depth_ft == native_depth
            && input.infiltration_depth_m == si_depth
            && (soil.native_rate_ft_per_s == 0.0
                || (native_depth > 0.0 && si_rate > 0.0 && si_depth > 0.0)),
        "soil source bridge identity",
        format!("receipt={soil:?}"),
    )?;
    let excess = (native_depth - native_available).max(0.0);
    require(
        soil.native_excess_depth_ft == excess,
        "native source excess",
        format!("stored={}, expected={excess}", soil.native_excess_depth_ft),
    )?;
    let expected_adjustment = (si_depth - available).max(0.0);
    require(
        input.adjustment_depth_m == expected_adjustment,
        "soil source adjustment",
        format!(
            "stored={}, expected={expected_adjustment}",
            input.adjustment_depth_m
        ),
    )?;
    require(
        expected_adjustment == 0.0 || available + expected_adjustment == si_depth,
        "soil source reconciliation",
        "positive adjustment must exactly reconcile returned infiltration",
    )?;
    if si_depth == 0.0 {
        return require(
            soil.native_roundoff_bound_ft == 0.0
                && soil.bridge_roundoff_bound_m == 0.0
                && soil.total_roundoff_bound_m == 0.0,
            "zero infiltration bounds",
            "zero infiltration requires zero bounds",
        );
    }
    let bridge = bridge_bound(
        dt,
        [a, dh, vh, ia, native_available],
        [
            input.incoming_depth_m,
            available,
            si_rate,
            si_depth,
            native_depth,
        ],
    )?;
    let total = upward_mul(METRES_PER_FOOT, soil.native_roundoff_bound_ft)
        .and_then(|native| upward_add(native, bridge))?;
    require(
        soil.bridge_roundoff_bound_m == bridge && soil.total_roundoff_bound_m == total,
        "soil source bound identity",
        format!(
            "bridge={}, expected_bridge={bridge}, total={}, expected_total={total}",
            soil.bridge_roundoff_bound_m, soil.total_roundoff_bound_m
        ),
    )?;
    for (name, value, bound) in [
        ("native source bound", excess, soil.native_roundoff_bound_ft),
        ("SI source bound", expected_adjustment, total),
    ] {
        if value > 0.0 {
            let upper = value + rounding_bound(value)?;
            require(
                upper.is_finite() && upper <= bound,
                name,
                format!("upper_excess={upper}, bound={bound}"),
            )?;
        }
    }
    Ok(())
}

fn bridge_bound(dt: f64, native: [f64; 5], si: [f64; 5]) -> Result<f64, SoilError> {
    let [a, dh, vh, ia, native_available] = native;
    let [incoming_depth, available, si_rate, si_depth, native_depth] = si;
    let input_native = sum(&[
        upward_mul(dt, rounding_bound(a)?)?,
        rounding_bound(dh)?,
        upward_mul(dt, rounding_bound(vh)?)?,
        upward_mul(dt, rounding_bound(ia)?)?,
        rounding_bound(native_available)?,
    ])?;
    let input = sum(&[
        upward_mul(METRES_PER_FOOT, input_native)?,
        rounding_bound(incoming_depth)?,
        rounding_bound(available)?,
    ])?;
    let output = sum(&[
        upward_mul(dt, rounding_bound(si_rate)?)?,
        rounding_bound(si_depth)?,
        upward_mul(METRES_PER_FOOT, rounding_bound(native_depth)?)?,
    ])?;
    Ok(upward_add(input, output)?)
}

fn sum(values: &[f64]) -> Result<f64, OutwardError> {
    values
        .iter()
        .try_fold(0.0, |total, &value| upward_add(total, value))
}

fn require(condition: bool, quantity: &str, context: impl AsRef<str>) -> Result<(), SoilError> {
    if condition {
        Ok(())
    } else {
        Err(SoilError::input(format!(
            "{quantity}: {}. Supply an unchanged source receipt",
            context.as_ref()
        )))
    }
}

fn nonnegative(values: &[f64], quantity: &str) -> Result<(), SoilError> {
    require(
        values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        quantity,
        format!("values={values:?}: require finite nonnegative values"),
    )
}
