//! Outward uncertainty and downward physical-throughput accounting.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::numerics::rounding::rounding_with_client;
use crate::runoff::ReservoirError;
use crate::soil::admission;
use crate::tensor::reservoir_storage::TensorReservoirContext;

pub(crate) fn check_nonnegative<R>(
    ctx: &TensorReservoirContext<R>,
    value: &Tensor<R>,
    name: &str,
) -> Result<(), ReservoirError>
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
        + TwoSumOps<R>,
{
    crate::numerics::reservoir::drainage::check_metadata(ctx, value, name)?;
    ctx.finite(value, name)?;
    ctx.reject_mask(
        &ctx.client.lt(value, &ctx.zero)?,
        name,
        &[("value", value)],
        &[],
        "Supply nonnegative finite component arithmetic",
    )
}

pub(crate) fn rounding<R>(
    ctx: &TensorReservoirContext<R>,
    value: &Tensor<R>,
) -> Result<Tensor<R>, ReservoirError>
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
        + TwoSumOps<R>,
{
    let bound = rounding_with_client(&ctx.client, value)?;
    ctx.finite(&bound, "arithmetic rounding bound")?;
    Ok(bound)
}

pub(crate) fn upward_add<R>(
    ctx: &TensorReservoirContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>, ReservoirError>
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
        + TwoSumOps<R>,
{
    check_nonnegative(ctx, a, "outward addition first operand")?;
    check_nonnegative(ctx, b, "outward addition second operand")?;
    let sum = ctx.client.add(a, b)?;
    ctx.finite(&sum, "outward addition intermediate")?;
    let result = ctx.client.add(&sum, &rounding(ctx, &sum)?)?;
    ctx.finite(&result, "outward addition")?;
    Ok(result)
}

pub(crate) fn upward_mul<R>(
    ctx: &TensorReservoirContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>, ReservoirError>
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
        + TwoSumOps<R>,
{
    check_nonnegative(ctx, a, "outward product first operand")?;
    check_nonnegative(ctx, b, "outward product second operand")?;
    let product = ctx.client.mul(a, b)?;
    ctx.finite(&product, "outward product intermediate")?;
    let result = ctx.client.add(&product, &rounding(ctx, &product)?)?;
    ctx.finite(&result, "outward product")?;
    Ok(result)
}

/// Nonnegative independent uncertainty charges, with exact zero identity.
pub(crate) fn add_charge(current: f64, step: f64) -> Result<f64, ReservoirError> {
    admission::add_adjustment_charge(current, step)
        .map_err(|e| ReservoirError::numerical(e.to_string()))
}

fn nonnegative(value: f64, name: &str) -> Result<f64, ReservoirError> {
    if value.is_finite() && value >= 0.0 {
        Ok(value)
    } else {
        Err(ReservoirError::numerical(format!(
            "{name}={value}: require nonnegative finite accounting"
        )))
    }
}

fn down(value: f64) -> Result<f64, ReservoirError> {
    nonnegative(value, "downward arithmetic intermediate")?;
    let bound =
        admission::rounding_bound(value).map_err(|e| ReservoirError::numerical(e.to_string()))?;
    Ok((value - bound).max(0.0))
}

/// Preserve each already-proved lower bound while physical throughput grows.
pub(crate) fn add_scale_lower(current: f64, step: f64) -> Result<f64, ReservoirError> {
    nonnegative(current, "accepted physical scale lower bound")?;
    nonnegative(step, "step physical scale lower bound")?;
    if step == 0.0 {
        return Ok(current);
    }
    if current == 0.0 {
        return Ok(step);
    }
    Ok(current.max(step).max(down(current + step)?))
}

pub(crate) fn permitted_budget(
    absolute: f64,
    relative: f64,
    scale_lower: f64,
) -> Result<f64, ReservoirError> {
    nonnegative(absolute, "absolute volume tolerance")?;
    nonnegative(relative, "relative volume tolerance")?;
    nonnegative(scale_lower, "physical scale lower bound")?;
    if relative == 0.0 || scale_lower == 0.0 {
        return Ok(absolute);
    }
    let product_lower = down(relative * scale_lower)?;
    if product_lower == 0.0 {
        return Ok(absolute);
    }
    if absolute == 0.0 {
        return Ok(product_lower);
    }
    Ok(absolute
        .max(product_lower)
        .max(down(absolute + product_lower)?))
}

/// A flat configured CPU reduction with the existing N+7 addition bound.
pub(crate) fn sum_lower<R>(
    ctx: &TensorReservoirContext<R>,
    values: &Tensor<R>,
) -> Result<f64, ReservoirError>
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
        + TwoSumOps<R>,
{
    check_nonnegative(ctx, values, "physical scale volumes")?;
    let count = values.numel();
    let sum = nonnegative(
        ctx.client.sum(values, &[], false)?.item::<f64>()?,
        "physical scale sum",
    )?;
    if sum == 0.0 || count == 1 {
        return Ok(sum);
    }
    let additions = count
        .checked_add(7)
        .ok_or_else(|| ReservoirError::input("physical scale reduction count overflows usize"))?;
    if additions as u128 > (1u128 << 53) {
        return Err(ReservoirError::input(
            "physical scale reduction requires N+7<=2^53",
        ));
    }
    let x = additions as f64 * f64::EPSILON;
    if !(0.0..1.0).contains(&x) {
        return Err(ReservoirError::input(
            "physical scale reduction requires (N+7)*EPSILON<1",
        ));
    }
    let factor = down(1.0 - x)?;
    if factor <= 0.0 {
        return Err(ReservoirError::input(
            "physical scale reduction requires positive denominator",
        ));
    }
    down(sum * factor)
}

/// Positive lane charges always consume the cumulative bound after reduction.
pub(crate) fn sum_upper<R>(
    ctx: &TensorReservoirContext<R>,
    values: &Tensor<R>,
) -> Result<f64, ReservoirError>
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
        + TwoSumOps<R>,
{
    check_nonnegative(ctx, values, "uncertainty charges")?;
    // Unit areas reuse the proved flat-reduction envelope without another algorithm.
    let (_, charge) = admission::AdjustmentReduction::new(values.numel())
        .and_then(|reduction| reduction.totals(&ctx.client, values, 1.0))
        .map_err(|e| ReservoirError::numerical(e.to_string()))?;
    Ok(charge)
}

/// Avoid charging zeros while retaining an outward result for positive lanes.
pub(crate) fn positive_charge<R>(
    ctx: &TensorReservoirContext<R>,
    active: &Tensor<R>,
    bound: &Tensor<R>,
) -> Result<Tensor<R>, ReservoirError>
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
        + TwoSumOps<R>,
{
    Ok(ctx.client.where_cond(active, bound, &ctx.zero)?)
}

/// Bound represented rainfall volume against the exact derived support area.
pub(crate) fn rain_area_bound<R>(
    ctx: &TensorReservoirContext<R>,
    rain_depth: &Tensor<R>,
    area_error: &Tensor<R>,
) -> Result<Tensor<R>, ReservoirError>
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
        + TwoSumOps<R>,
{
    check_nonnegative(ctx, rain_depth, "rain depth")?;
    check_nonnegative(ctx, area_error, "area rounding bound")?;
    let volume = ctx.client.mul(&ctx.area, rain_depth)?;
    ctx.finite(&volume, "rain volume")?;
    let bound = upward_add(
        ctx,
        &upward_mul(ctx, rain_depth, area_error)?,
        &rounding(ctx, &volume)?,
    )?;
    positive_charge(ctx, &ctx.client.gt(rain_depth, &ctx.zero)?, &bound)
}

#[cfg(test)]
mod tests;
