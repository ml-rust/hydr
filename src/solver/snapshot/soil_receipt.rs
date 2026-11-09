//! Algebraic native/SI admission receipts, without replaying antecedent soil memory.

use super::check::{nonnegative, rejected, require};
use crate::soil::SoilError;
use crate::soil::admission::{rounding_bound, upward_add, upward_mul};
use crate::solver::{EngineError, SoilSourceReceipt, SourceReceipt};

const METRES_PER_FOOT: f64 = 0.3048;

pub(super) fn check(
    receipt: &SourceReceipt,
    soil: &SoilSourceReceipt,
    rain: f64,
    available: f64,
    dt: f64,
) -> Result<(), EngineError> {
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
    let a = rain / METRES_PER_FOOT;
    let dh = receipt.ponded_depth_m / METRES_PER_FOOT;
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
            && receipt.infiltration_depth_m == si_depth
            && (soil.native_rate_ft_per_s == 0.0
                || (native_depth > 0.0 && si_rate > 0.0 && si_depth > 0.0)),
        "soil source bridge identity",
        format!("cell={}: receipt={soil:?}", receipt.cell),
    )?;
    let excess = (native_depth - native_available).max(0.0);
    require(
        soil.native_excess_depth_ft == excess,
        "native source excess",
        format!(
            "cell={}: stored={}, expected={excess}",
            receipt.cell, soil.native_excess_depth_ft
        ),
    )?;
    let expected_adjustment = (si_depth - available).max(0.0);
    require(
        receipt.water_adjustment_depth_m == expected_adjustment,
        "soil source adjustment",
        format!(
            "cell={}: stored={}, expected={expected_adjustment}",
            receipt.cell, receipt.water_adjustment_depth_m
        ),
    )?;
    if si_depth == 0.0 {
        return require(
            soil.native_roundoff_bound_ft == 0.0
                && soil.bridge_roundoff_bound_m == 0.0
                && soil.total_roundoff_bound_m == 0.0,
            "zero infiltration bounds",
            format!(
                "cell={}: zero infiltration requires zero bounds",
                receipt.cell
            ),
        );
    }
    let bridge = bridge_bound(
        dt,
        [a, dh, vh, ia, native_available],
        [
            receipt.rain_depth_m,
            available,
            si_rate,
            si_depth,
            native_depth,
        ],
    )
    .map_err(|error| rejected("soil bridge bound", error.to_string()))?;
    let total = upward_mul(METRES_PER_FOOT, soil.native_roundoff_bound_ft)
        .and_then(|native| upward_add(native, bridge))
        .map_err(|error| rejected("soil total bound", error.to_string()))?;
    require(
        soil.bridge_roundoff_bound_m == bridge && soil.total_roundoff_bound_m == total,
        "soil source bound identity",
        format!(
            "cell={}: bridge={}, expected_bridge={bridge}, total={}, expected_total={total}",
            receipt.cell, soil.bridge_roundoff_bound_m, soil.total_roundoff_bound_m
        ),
    )?;
    for (name, value, bound) in [
        ("native source bound", excess, soil.native_roundoff_bound_ft),
        ("SI source bound", expected_adjustment, total),
    ] {
        if value > 0.0 {
            let upper =
                value + rounding_bound(value).map_err(|error| rejected(name, error.to_string()))?;
            require(
                upper.is_finite() && upper <= bound,
                name,
                format!("cell={}: upper_excess={upper}, bound={bound}", receipt.cell),
            )?;
        }
    }
    Ok(())
}

fn bridge_bound(dt: f64, native: [f64; 5], si: [f64; 5]) -> Result<f64, SoilError> {
    let [a, dh, vh, ia, native_available] = native;
    let [rain_depth, available, si_rate, si_depth, native_depth] = si;
    let input_native = sum(&[
        upward_mul(dt, rounding_bound(a)?)?,
        rounding_bound(dh)?,
        upward_mul(dt, rounding_bound(vh)?)?,
        upward_mul(dt, rounding_bound(ia)?)?,
        rounding_bound(native_available)?,
    ])?;
    let input = sum(&[
        upward_mul(METRES_PER_FOOT, input_native)?,
        rounding_bound(rain_depth)?,
        rounding_bound(available)?,
    ])?;
    let output = sum(&[
        upward_mul(dt, rounding_bound(si_rate)?)?,
        rounding_bound(si_depth)?,
        upward_mul(METRES_PER_FOOT, rounding_bound(native_depth)?)?,
    ])?;
    upward_add(input, output)
}

fn sum(values: &[f64]) -> Result<f64, SoilError> {
    values
        .iter()
        .try_fold(0.0, |total, &value| upward_add(total, value))
}
