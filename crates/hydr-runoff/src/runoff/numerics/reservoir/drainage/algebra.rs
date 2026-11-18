//! Represented reservoir law, geometry products, and root reconstruction.

use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::budget::DrainageBudget;
use crate::runoff::ReservoirError;
use crate::runoff::tensor::reservoir_storage::TensorReservoirContext;
use numr::dtype::DType;
use numr::runtime::Device;

pub(crate) const EXPONENT: f64 = 1.6666667;

pub(crate) fn check_metadata<R>(
    context: &TensorReservoirContext<R>,
    value: &Tensor<R>,
    quantity: &str,
) -> Result<(), ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    if value.shape() != context.zero.shape()
        || value.dtype() != DType::F64
        || !value.device().is_same(context.zero.device())
    {
        return Err(ReservoirError::input(format!(
            "{quantity}: shape={:?}, dtype={:?}: require the configured runtime F64 shape {:?} on the matching device",
            value.shape(),
            value.dtype(),
            context.zero.shape()
        )));
    }
    Ok(())
}

pub(crate) fn volume<R>(
    context: &TensorReservoirContext<R>,
    depth: &Tensor<R>,
    quantity: &str,
) -> Result<Tensor<R>, ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let c = &context.client;
    let result = c.mul(&context.area, depth)?;
    context.finite(&result, quantity)?;
    context.reject_mask(
        &c.mul(&c.gt(depth, &context.zero)?, &c.eq(&result, &context.zero)?)?,
        quantity,
        &[("value", depth), ("area", &context.area)],
        &[],
        "Positive area product underflows. Rescale geometry or state",
    )?;
    Ok(result)
}

pub(super) fn law<R>(
    context: &TensorReservoirContext<R>,
    excess: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<Tensor<R>, ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let c = &context.client;
    let x = c.where_cond(active, excess, &context.zero)?;
    let power = c.cr_pow_scalar(&x, EXPONENT)?;
    context.finite(&power, "reservoir excess power")?;
    let alpha = c.where_cond(active, &context.alpha, &context.zero)?;
    let rate = c.mul(&alpha, &power)?;
    context.finite(&rate, "reservoir endpoint rate")?;
    context.reject_mask(
        &c.mul(&c.gt(&x, &context.zero)?, &c.eq(&rate, &context.zero)?)?,
        "reservoir endpoint law",
        &[("excess", &x), ("power", &power), ("coefficient", &alpha)],
        &[],
        "Positive drainage rate underflows. Rescale the reservoir",
    )?;
    Ok(rate)
}

pub(super) fn residual<R>(
    context: &TensorReservoirContext<R>,
    x: &Tensor<R>,
    prepared: &DrainageBudget<'_, R>,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let c = &context.client;
    let x = c.where_cond(prepared.active(), x, &context.zero)?;
    let alpha = c.where_cond(prepared.active(), &context.alpha, &context.zero)?;
    let rate = c.mul(&alpha, &c.cr_pow_scalar(&x, EXPONENT)?)?;
    let loss = c.mul(prepared.active_duration(), &rate)?;
    c.where_cond(
        prepared.active(),
        &c.sub(&c.add(&x, &loss)?, prepared.excess())?,
        &context.zero,
    )
}

pub(super) fn candidate_depth<R>(
    context: &TensorReservoirContext<R>,
    prepared: &DrainageBudget<'_, R>,
    upper: &Tensor<R>,
) -> Result<Tensor<R>, ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let c = &context.client;
    let algebraic = c.where_cond(
        &context.instantaneous,
        &c.minimum(prepared.available_depth(), &context.depression)?,
        prepared.available_depth(),
    )?;
    Ok(c.where_cond(
        prepared.active(),
        &c.add(&context.depression, upper)?,
        &algebraic,
    )?)
}
