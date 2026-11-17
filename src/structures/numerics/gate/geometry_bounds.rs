//! Independent gate-expression uncertainty and resolved-positive operand admission.
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::ops::ScalarOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::structures::tensor::TensorStructureContext;
use numr::tensor::Tensor;

const QUANTITY: &str = "gate geometry uncertainty bound overflows";

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
    super::super::bounds::add(context, a, b, QUANTITY)
}

pub(super) fn resolved_positive<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    mask: &Tensor<R>,
    value: &Tensor<R>,
    error: &Tensor<R>,
) -> Result<(), StructureError>
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
    let center = c.where_cond(mask, value, &context.one)?;
    let bound = c.where_cond(mask, error, &context.zero)?;
    context.positive(
        &center,
        "active positive gate geometry is unresolved or nonfinite",
    )?;
    context.finite(&bound, QUANTITY)?;
    context.reject(
        &c.lt(&bound, &context.zero)?,
        "gate geometry bound must remain nonnegative",
    )?;
    let ratio = super::super::bounds::divide(context, &bound, &center, QUANTITY)?;
    context.reject(
        &c.gt(&ratio, &c.add_scalar(&context.zero, 64.0 * f64::EPSILON)?)?,
        "active gate geometry uncertainty exceeds 64*EPSILON relative admission",
    )
}
