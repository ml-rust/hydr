//! Hydraulic opening support and explicit absolute-or-relative receipt admission.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::tensor::structures::{
    TensorStructureContext,
    gate::trajectory::{TensorTrajectoryContext, TensorTrajectoryInput},
};
use numr::tensor::Tensor;

pub(super) fn check<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    trajectory: &TensorTrajectoryContext<R>,
    input: &TensorTrajectoryInput<R>,
    opening: &Tensor<R>,
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
    if context.gate.is_none() {
        return Err(StructureError::numerical(
            "bound trajectory context lacks gate tensors",
        ));
    }
    context.finite(opening, "trajectory opening")?;
    context.finite(error, "trajectory absolute opening error receipt")?;
    context.reject(
        &c.lt(error, &context.zero)?,
        "trajectory error receipt is negative",
    )?;
    let outside = c.maximum(
        &c.lt(opening, &input.minimum)?,
        &c.maximum(
            &c.gt(opening, &input.maximum)?,
            &c.gt(opening, &trajectory.maximum)?,
        )?,
    )?;
    context.reject(
        &c.mul(&trajectory.assigned_mask, &outside)?,
        "trajectory opening exceeds original segment or gate range",
    )?;
    let positive = c.sub(&trajectory.assigned_mask, &input.closed)?;
    context.reject(
        &c.mul(&positive, &c.le(opening, &context.zero)?)?,
        "original positive trajectory support returns zero opening",
    )?;
    context.reject(
        &c.mul(
            &input.closed,
            &c.maximum(&c.ne(opening, &context.zero)?, &c.ne(error, &context.zero)?)?,
        )?,
        "declared closed trajectory support lacks exact zero value and receipt",
    )?;
    let absolute = c.le(error, &trajectory.absolute_limit)?;
    let need_relative = c.mul(&positive, &c.sub(&context.one, &absolute)?)?;
    let relative_active = c.mul(
        &need_relative,
        &c.gt(&trajectory.relative_limit, &context.zero)?,
    )?;
    let numerator = c.where_cond(&relative_active, error, &context.zero)?;
    let denominator = c.where_cond(&relative_active, opening, &context.one)?;
    let relative = crate::numerics::structures::bounds::divide(
        context,
        &numerator,
        &denominator,
        "trajectory relative opening error",
    )?;
    let relative_admitted = c.mul(
        &relative_active,
        &c.le(&relative, &trajectory.relative_limit)?,
    )?;
    context.reject(
        &c.mul(
            &trajectory.assigned_mask,
            &c.sub(&context.one, &c.maximum(&absolute, &relative_admitted)?)?,
        )?,
        "trajectory opening error exceeds both explicit accuracy limits",
    )
}
