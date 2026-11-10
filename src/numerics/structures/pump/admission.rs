//! Running wet support, strict suction, and declared speed admission before curve queries.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::{StructureError, StructureInputErrorKind};
use crate::tensor::structures::{
    TensorStructureContext, TensorStructureInput,
    pump::{TensorPumpContext, TensorPumpInput},
};
use numr::tensor::Tensor;

pub(super) fn heads<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    pump: &TensorPumpContext<R>,
    input: &TensorStructureInput<R>,
    operation: &TensorPumpInput<R>,
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
    let forward = c.gt(&pump.orientation, &context.zero)?;
    let donor = c.where_cond(&forward, &input.stage_a, &input.stage_b)?;
    let receiver = c.where_cond(&forward, &input.stage_b, &input.stage_a)?;
    let wet = c.mul(&input.wet_a, &input.wet_b)?;
    context.reject_support(
        &c.mul(&operation.running, &c.sub(&context.one, &wet)?)?,
        "running pump requires known wet donor and receiving support; dry delivery is unsupported",
    )?;
    context.reject_support(&c.mul(&operation.running,&c.le(&donor,&pump.minimum_suction)?)?,"running pump donor must lie strictly above its declared minimum suction stage, including shutoff")?;
    let invalid_speed = c.maximum(
        &c.lt(&operation.speed, &pump.minimum_speed)?,
        &c.gt(&operation.speed, &pump.maximum_speed)?,
    )?;
    context.reject_input(
        &c.mul(&operation.running, &invalid_speed)?,
        StructureInputErrorKind::OperatingPoint,
        "running speed ratio lies outside its immutable reference/affinity policy",
    )?;
    Ok((
        c.where_cond(&operation.running, &donor, &context.zero)?,
        c.where_cond(&operation.running, &receiver, &context.zero)?,
    ))
}
