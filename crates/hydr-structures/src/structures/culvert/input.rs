//! Query admission: shape, dtype, device and value checks with one scalar host read.
//!
//! A device mask counts the bad rows. Only a rejected call copies the inputs to the host
//! to name the first offending row.

use crate::structures::numerics::culvert::BarrelClient;
use crate::structures::tensor::culvert::TensorBarrelContext;
use crate::structures::{CulvertBarrel, StructureError, StructureInputErrorKind};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ReduceOps};
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;

/// Rejects a tensor that is not F64 of shape `[m]` on the barrel device.
pub(super) fn check_tensor<R>(
    context: &TensorBarrelContext<R>,
    name: &str,
    value: &Tensor<R>,
) -> Result<(), StructureError>
where
    R: Runtime<DType = DType>,
{
    let m = context.width();
    if value.shape() != [m] {
        return Err(StructureError::input(
            StructureInputErrorKind::OperatingPoint,
            format!(
                "{name} shape={:?},barrels={m}: supply one value per barrel in a [{m}] tensor",
                value.shape()
            ),
        ));
    }
    if value.dtype() != DType::F64 {
        return Err(StructureError::input(
            StructureInputErrorKind::OperatingPoint,
            format!("{name} dtype={:?}: supply an F64 tensor", value.dtype()),
        ));
    }
    if !context.device.is_same(value.device()) {
        return Err(StructureError::input(
            StructureInputErrorKind::OperatingPoint,
            format!("{name} uses another device: supply tensors on the barrel batch device"),
        ));
    }
    Ok(())
}

/// What a depth value measures, for rejection messages.
const DEPTH: &str = "depth above the invert in metres";

/// What a discharge value measures, for rejection messages.
const DISCHARGE: &str = "discharge per barrel in cubic metres per second";

/// Rejects a depth that is negative or nonfinite, naming the first such row.
pub(super) fn check_depths<R>(
    context: &TensorBarrelContext<R>,
    name: &str,
    depth: &Tensor<R>,
) -> Result<(), StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    check_nonnegative(context, name, depth, DEPTH)
}

/// Rejects a discharge that is negative or nonfinite, naming the first such row.
pub(super) fn check_discharges<R>(
    context: &TensorBarrelContext<R>,
    name: &str,
    discharge: &Tensor<R>,
) -> Result<(), StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    check_nonnegative(context, name, discharge, DISCHARGE)
}

/// Rejects a gravity that is nonfinite or not positive.
pub(super) fn check_gravity(gravity: f64) -> Result<(), StructureError> {
    if gravity.is_finite() && gravity > 0.0 {
        return Ok(());
    }
    Err(StructureError::input(
        StructureInputErrorKind::OperatingPoint,
        format!(
            "gravity_m_per_s2={gravity:?}: supply a finite positive gravity in metres per second squared"
        ),
    ))
}

fn check_nonnegative<R>(
    context: &TensorBarrelContext<R>,
    name: &str,
    value: &Tensor<R>,
    what: &str,
) -> Result<(), StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    check_tensor(context, name, value)?;
    let c = &context.client;
    let infinite = context.constant(f64::INFINITY)?;
    let good = c.mul(&c.ge(value, &context.zero)?, &c.lt(value, &infinite)?)?;
    if count_bad(context, &good)? == 0.0 {
        return Ok(());
    }
    let values = value.try_to_vec::<f64>()?;
    for (row, &value) in values.iter().enumerate() {
        nonnegative_value(name, value, what).map_err(|error| error.at(row))?;
    }
    Err(rows_disagree())
}

fn nonnegative_value(name: &str, value: f64, what: &str) -> Result<(), StructureError> {
    if !value.is_finite() {
        return Err(StructureError::input(
            StructureInputErrorKind::NonfiniteInput,
            format!("{name}={value:?}: supply a finite {what}"),
        ));
    }
    if value < 0.0 {
        return Err(StructureError::input(
            StructureInputErrorKind::OperatingPoint,
            format!("{name}={value:?}: supply a nonnegative {what}"),
        ));
    }
    Ok(())
}

/// Rejects a depth range that is not `0 <= lower <= upper < D`, naming the first such row.
pub(super) fn check_range<R>(
    barrels: &[CulvertBarrel],
    context: &TensorBarrelContext<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> Result<(), StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    check_tensor(context, "lower_m", lower)?;
    check_tensor(context, "upper_m", upper)?;
    let c = &context.client;
    // 0 <= lower <= upper < D also excludes NaN and infinite ends.
    let ordered = c.mul(&c.ge(lower, &context.zero)?, &c.le(lower, upper)?)?;
    let good = c.mul(&ordered, &c.lt(upper, &context.rise)?)?;
    if count_bad(context, &good)? == 0.0 {
        return Ok(());
    }
    let lows = lower.try_to_vec::<f64>()?;
    let highs = upper.try_to_vec::<f64>()?;
    for (row, (barrel, (&low, &high))) in barrels.iter().zip(lows.iter().zip(&highs)).enumerate() {
        for (name, value) in [("lower_m", low), ("upper_m", high)] {
            nonnegative_value(name, value, DEPTH).map_err(|error| error.at(row))?;
        }
        if low > high {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                format!("lower_m={low:?},upper_m={high:?}: supply lower_m <= upper_m"),
            )
            .at(row));
        }
        let rise = barrel.rise_m();
        if high >= rise {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                format!(
                    "upper_m={high:?},rise_m={rise:?}: the range reaches the crown. Split it at the crown and use full_geometry above it"
                ),
            )
            .at(row));
        }
    }
    Err(rows_disagree())
}

fn count_bad<R>(context: &TensorBarrelContext<R>, good: &Tensor<R>) -> Result<f64, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = &context.client;
    let bad = c.sub(&context.one, good)?;
    Ok(c.sum(&bad, &[0], false)?.item::<f64>()?)
}

fn rows_disagree() -> StructureError {
    StructureError::numerical(
        "the device check found a rejected row that the host copy does not show: report this runtime",
    )
}
