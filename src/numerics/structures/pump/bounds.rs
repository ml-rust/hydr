//! Source-specific outward head-normalization bounds with exact zero identities.
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
    super::super::bounds::rounding(
        context,
        value,
        "pump normalization uncertainty bound overflows",
    )
}

pub(super) fn add<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
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
    super::super::bounds::add(
        context,
        a,
        b,
        "pump normalization uncertainty bound overflows",
    )
}

pub(super) fn divide<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    numerator: &Tensor<R>,
    denominator: &Tensor<R>,
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
    super::super::bounds::divide(
        context,
        numerator,
        denominator,
        "pump normalization uncertainty bound overflows",
    )
}

pub(super) fn interval<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    query: &Tensor<R>,
    bound: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>), StructureError>
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
    let has_bound = c.gt(bound, &context.zero)?;
    let center = c.where_cond(&has_bound, query, &context.zero)?;
    let lo0 = c.sub(&center, bound)?;
    let hi0 = c.add(&center, bound)?;
    context.finite(&lo0, "pump normalization lower interval overflows")?;
    context.finite(&hi0, "pump normalization upper interval overflows")?;
    let lower = c.sub(&lo0, &rounding(context, &lo0)?)?;
    let upper = c.add(&hi0, &rounding(context, &hi0)?)?;
    context.finite(&lower, "pump normalization lower interval overflows")?;
    context.finite(&upper, "pump normalization upper interval overflows")?;
    Ok((
        c.where_cond(&has_bound, &lower, query)?,
        c.where_cond(&has_bound, &upper, query)?,
    ))
}
