//! Shape-neutral outward motion arithmetic without global rejection of optional certificates.
use numr::ops::BinaryOps;
use numr::ops::ScalarOps;
use numr::ops::TypeConversionOps;
use numr::ops::UnaryOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::tensor::structures::TensorStructureContext;
use numr::tensor::Tensor;

pub(super) fn finite<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    value: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.client;
    let invalid = c.maximum(
        &c.cast(&c.isnan(value)?, value.dtype())?,
        &c.cast(&c.isinf(value)?, value.dtype())?,
    )?;
    Ok(c.add_scalar(&c.neg(&invalid)?, 1.0)?)
}
pub(super) fn up<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    value: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let bound = crate::numerics::rounding::rounding_with_client(&context.client, value)?;
    Ok(context.client.add(value, &bound)?)
}
pub(super) fn down<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    value: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let bound = crate::numerics::rounding::rounding_with_client(&context.client, value)?;
    Ok(context.client.sub(value, &bound)?)
}
