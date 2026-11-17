//! Shared outward arithmetic for independent structure-expression witnesses.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::tensor::structures::TensorStructureContext;
use numr::tensor::Tensor;

pub(super) fn rounding<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    value: &Tensor<R>,
    quantity: &str,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let bound = crate::numerics::rounding::rounding_with_client(&context.client, value)?;
    context.finite(&bound, quantity)?;
    Ok(bound)
}

pub(super) fn add<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
    quantity: &str,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.client;
    let a_zero = c.eq(a, &context.zero)?;
    let b_zero = c.eq(b, &context.zero)?;
    let both = c.mul(&c.gt(a, &context.zero)?, &c.gt(b, &context.zero)?)?;
    let sum = c.add(a, b)?;
    context.finite(&sum, quantity)?;
    let used = c.where_cond(&both, &sum, &context.zero)?;
    let padded = c.add(&used, &rounding(context, &used, quantity)?)?;
    let bound = c.where_cond(&a_zero, b, &c.where_cond(&b_zero, a, &padded)?)?;
    context.finite(&bound, quantity)?;
    Ok(bound)
}

pub(super) fn divide<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    numerator: &Tensor<R>,
    denominator: &Tensor<R>,
    quantity: &str,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.client;
    let quotient = c.div(numerator, denominator)?;
    context.finite(&quotient, quantity)?;
    let padded = c.add(&quotient, &rounding(context, &quotient, quantity)?)?;
    let bound = c.where_cond(&c.eq(numerator, &context.zero)?, &context.zero, &padded)?;
    context.finite(&bound, quantity)?;
    Ok(bound)
}
