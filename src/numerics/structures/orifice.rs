//! All-log fully submerged orifice evaluation without unnecessary physical products.
use numr::ops::BinaryOps;
use numr::ops::ConditionalOps;
use numr::ops::ScalarOps;
use numr::ops::UnaryOps;
use numr::{dtype::DType, runtime::Runtime};

use super::direction::Classification;
use crate::structures::StructureError;
use crate::tensor::structures::TensorStructureContext;
use numr::tensor::Tensor;

pub(super) struct OrificeResult<R: Runtime<DType = DType>> {
    pub magnitude: Tensor<R>,
    pub logarithm: Tensor<R>,
    pub gap: Tensor<R>,
}

pub(super) fn exp_active<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    active: &Tensor<R>,
    raw: &Tensor<R>,
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
    let logarithm = c.where_cond(active, raw, &context.zero)?;
    context.finite(&logarithm, "active log discharge")?;
    let magnitude = c.exp(&logarithm)?;
    context.positive(
        &magnitude,
        "active final discharge overflows or underflows to zero",
    )?;
    Ok((c.where_cond(active, &magnitude, &context.zero)?, logarithm))
}

pub(super) fn evaluate<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    class: &Classification<R>,
) -> Result<OrificeResult<R>, StructureError>
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
    let high = c.where_cond(&class.orifice, &class.high, &context.one)?;
    let low = c.where_cond(&class.orifice, &class.low, &context.zero)?;
    let gap = c.sub(&high, &low)?;
    context.positive(
        &gap,
        "active original orifice head gap must remain positive finite",
    )?;
    let root_log = c.mul_scalar(
        &c.add(
            &c.add(&context.log_two, &context.log_gravity)?,
            &c.log(&gap)?,
        )?,
        0.5,
    )?;
    let raw = c.add(&context.log_orifice_scale, &root_log)?;
    let (magnitude, logarithm) = exp_active(context, &class.orifice, &raw)?;
    Ok(OrificeResult {
        magnitude,
        logarithm,
        gap: c.where_cond(&class.orifice, &gap, &context.zero)?,
    })
}
