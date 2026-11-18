//! Running wet or free-outlet support, strict suction, and declared speed admission before curve
//! queries.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::tensor::{
    TensorStructureContext, TensorStructureInput,
    pump::{TensorPumpContext, TensorPumpInput},
};
use crate::structures::{StructureError, StructureInputErrorKind};
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
    let donor_wet = c.where_cond(&forward, &input.wet_a, &input.wet_b)?;
    let receiver_wet = c.where_cond(&forward, &input.wet_b, &input.wet_a)?;
    let receiver_bed = c.where_cond(&forward, &input.bed_b, &input.bed_a)?;
    let delivered = c.maximum(&receiver_wet, &pump.has_outlet)?;
    context.reject_support(
        &c.mul(
            &operation.running,
            &c.sub(&context.one, &c.mul(&donor_wet, &delivered)?)?,
        )?,
        "running pump requires a known wet donor and a known wet receiver or a declared free outlet; dry delivery without an outlet is unsupported",
    )?;
    let receiver_dry = c.sub(&context.one, &receiver_wet)?;
    context.reject_support(
        &c.mul(
            &c.mul(&operation.running, &receiver_dry)?,
            &c.gt(&receiver_bed, &pump.outlet)?,
        )?,
        "running pump dry receiver bed lies above its declared free outlet; raise the outlet elevation or supply a wet receiver",
    )?;
    context.reject_support(&c.mul(&operation.running,&c.le(&donor,&pump.minimum_suction)?)?,"running pump donor must lie strictly above its declared minimum suction stage, including shutoff")?;
    // A free outlet lifts to `max(receiver, outlet)`, the bed standing in for a dry receiver's
    // stage. Pumps without an outlet keep the receiving stage bit for bit.
    let level = c.where_cond(&receiver_wet, &receiver, &receiver_bed)?;
    let receiver = c.where_cond(
        &pump.has_outlet,
        &c.maximum(&level, &pump.outlet)?,
        &receiver,
    )?;
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
