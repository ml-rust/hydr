//! Stable all-log suppressed rectangular weir laws with direct original-stage gap.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::ops::CorrectlyRoundedOps;
use numr::ops::ScalarOps;
use numr::ops::UnaryOps;
use numr::{dtype::DType, runtime::Runtime};

use super::direction::Classification;
use crate::structures::StructureError;
use crate::structures::tensor::TensorStructureContext;
use numr::tensor::Tensor;

pub(super) const VILLEMONTE_EXPONENT: f64 = 0.385;

pub(super) struct WeirResult<R: Runtime<DType = DType>> {
    pub magnitude: Tensor<R>,
    pub logarithm: Tensor<R>,
    pub upstream: Tensor<R>,
    pub downstream: Tensor<R>,
    pub gap: Tensor<R>,
    pub ratio: Tensor<R>,
    pub k: Tensor<R>,
}

pub(super) fn free<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    class: &Classification<R>,
) -> Result<WeirResult<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::CorrectlyRoundedOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.client;
    let active = &class.free_weir;
    let high = c.where_cond(active, &class.high, &context.one)?;
    let crest = c.where_cond(active, &context.crest, &context.zero)?;
    let hu = c.sub(&high, &crest)?;
    context.positive(
        &hu,
        "active free-overflow head above crest must remain positive finite",
    )?;
    let raw = c.add(
        &context.log_weir_scale,
        &c.mul_scalar(&c.cr_log(&hu)?, 1.5)?,
    )?;
    let (magnitude, logarithm) = super::orifice::exp_active(context, active, &raw)?;
    Ok(WeirResult {
        magnitude,
        logarithm,
        upstream: c.where_cond(active, &hu, &context.zero)?,
        downstream: context.zero.clone(),
        gap: context.zero.clone(),
        ratio: context.zero.clone(),
        k: context.zero.clone(),
    })
}

pub(super) fn submerged<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    class: &Classification<R>,
) -> Result<WeirResult<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::CorrectlyRoundedOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.client;
    let active = &class.submerged_weir;
    let high = c.where_cond(active, &class.high, &context.one)?;
    let low = c.where_cond(active, &class.low, &c.mul_scalar(&context.one, 0.5)?)?;
    let crest = c.where_cond(active, &context.crest, &context.zero)?;
    let hu = c.sub(&high, &crest)?;
    let hd = c.sub(&low, &crest)?;
    let gap = c.sub(&high, &low)?;
    context.positive(
        &hu,
        "active submerged upstream head must remain positive finite",
    )?;
    context.positive(
        &hd,
        "active submerged downstream head must remain positive finite",
    )?;
    context.positive(
        &gap,
        "active direct original-stage gap must remain positive finite",
    )?;
    let ratio = c.div(&hd, &hu)?;
    context.finite(&ratio, "represented submerged head ratio")?;
    context.reject(
        &c.maximum(&c.lt(&ratio, &context.zero)?, &c.gt(&ratio, &context.one)?)?,
        "represented submerged head ratio lies outside [0,1]",
    )?;
    // No outward bound covers this square root, so no runtime square-root contract is checked.
    // Its error reaches only `k`, which feeds the correctly rounded log/exp path to the discharge.
    // Downstream finite and positive checks reject any nonfinite or vanished result.
    let k = c.add(
        &context.one,
        &c.div(&ratio, &c.add(&context.one, &c.sqrt(&ratio)?)?)?,
    )?;
    let raw = c.add(
        &c.add(
            &c.add(
                &context.log_weir_scale,
                &c.mul_scalar(&c.cr_log(&hu)?, 1.5 - VILLEMONTE_EXPONENT)?,
            )?,
            &c.mul_scalar(&c.cr_log(&gap)?, VILLEMONTE_EXPONENT)?,
        )?,
        &c.mul_scalar(&c.cr_log(&k)?, VILLEMONTE_EXPONENT)?,
    )?;
    let (magnitude, logarithm) = super::orifice::exp_active(context, active, &raw)?;
    Ok(WeirResult {
        magnitude,
        logarithm,
        upstream: c.where_cond(active, &hu, &context.zero)?,
        downstream: c.where_cond(active, &hd, &context.zero)?,
        gap: c.where_cond(active, &gap, &context.zero)?,
        ratio: c.where_cond(active, &ratio, &context.zero)?,
        k: c.where_cond(active, &k, &context.zero)?,
    })
}
